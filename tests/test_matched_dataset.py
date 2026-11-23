"""Phase 4 integration and historical regression tests; no model training."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.cosmology import Planck18
import astropy.units as u
from torch.utils.data import DataLoader

from clusterprep import ClusterDataset
from clusterprep.matched_dataset import MatchedResolutionConfig, CoverageEligibilityError
from clusterprep.matched_resolution import prepare_matched_pair
from clusterprep.modalities import fit_normalization
from clusterprep.preprocessing import preprocess, normalize
from test_matched_resolution import grid


@pytest.fixture
def catalog(tmp_path):
    cid = 'synthetic+paired'
    w = grid(256)
    yy, xx = np.indices((256, 256))
    base = np.exp(-((xx-128)**2+(yy-128)**2)/200) + .01*np.sin(xx/4)
    paths = {}
    for name, beam, data in [('RAW', (6, 3, 32), base.copy()),
                             ('T50kpcSUB', (9, 5, 117), base.copy()*2),
                             ('CHANDRA', (6, 3, 0), base.copy()+.1)]:
        header = w.to_header()
        header.update(BMAJ=beam[0]/3600, BMIN=beam[1]/3600, BPA=beam[2], BUNIT='Jy/beam')
        if name == 'RAW':
            data[100:114, 102:118] = np.nan
        if name == 'T50kpcSUB':
            data[140:153, 150:163] = np.nan
        if name == 'CHANDRA':
            data[:, :116] = np.nan
        path = tmp_path/f'{name}.fits'
        fits.writeto(path, data.astype('float32'), header)
        paths[name] = str(path)
    record = dict(cluster_id=cid, ra_deg=170., dec_deg=30., redshift=.2, r500_kpc=100.)
    info = {cid: dict(cluster_id=cid, name=cid, raw_path=paths['RAW'],
                     radio_paths={k: v for k, v in paths.items() if k != 'CHANDRA'},
                     chandra_path=paths['CHANDRA'], labels={'RR', 'RH'}, physical_record=record)}
    return info, cid, tmp_path/'processed'


def dataset(catalog, **kwargs):
    info, cid, root = catalog
    defaults = dict(mode='pair', task='rr_rh', size=128, processed_dir=root,
                    radio_preprocessing='matched_resolution')
    return ClusterDataset(info, [cid], ['RR', 'RH'], **(defaults | kwargs))


@pytest.mark.parametrize('mode,modalities', [('raw', None), ('pair', None), ('multimodal', None),
                                             ('raw', ['radio', 'xray']), ('raw', ['xray', 'radio'])])
def test_shared_chandra_masks_shapes_labels_and_batch(catalog, mode, modalities):
    raw = dataset(catalog, mode=mode, modalities=modalities, radio_source='RAW')
    tapered = dataset(catalog, mode=mode, modalities=modalities, radio_source='T50kpcSUB')
    a, b = raw[0], tapered[0]
    assert a['raw'].shape == (1, 128, 128) and a['raw'].dtype == torch.float32
    assert torch.equal(a['raw_mask'], b['raw_mask'])
    assert torch.equal(a['label'], torch.tensor([1., 1.]))
    assert torch.equal(a['label_mask'], b['label_mask'])
    assert torch.isfinite(a['raw']).all()
    assert torch.all(a['raw'][~a['raw_mask'].bool()] == 0)
    if mode != 'raw' or modalities is not None:
        assert torch.equal(a['chandra'], b['chandra'])
        assert torch.equal(a['chandra_mask'], b['chandra_mask'])
    if modalities is not None:
        assert a['image'].shape == (3, 128, 128)  # unchanged model channel layout
    meta_a, meta_b = raw.get_resolution_metadata(0), tapered.get_resolution_metadata(0)
    assert meta_a['cache_key'] == meta_b['cache_key']
    assert meta_a['chandra_sha256'] == meta_b['chandra_sha256']
    assert meta_a['chandra_mask_sha256'] == meta_b['chandra_mask_sha256']
    assert meta_a['coverage'] == meta_b['coverage']
    assert set(meta_a['coverage']) == {'RAW', 'T50kpcSUB', 'CHANDRA', 'intersection'}
    assert meta_a['selected_radio_coverage'] == meta_a['coverage']['intersection']
    batch = next(iter(DataLoader(raw, batch_size=1)))
    assert batch['raw'].shape == (1, 1, 128, 128)
    assert batch['resolution_metadata']['eligible'].item()
    assert len(list(catalog[2].rglob('*.npz'))) == 1
    assert not list(catalog[2].rglob('*.fits'))


def test_specific_masks_and_physical_beam(catalog):
    a = dataset(catalog, matched_resolution={'radio_mask': 'specific'})
    b = dataset(catalog, radio_source='T50kpcSUB', matched_resolution={'radio_mask': 'specific'})
    aa, bb = a[0], b[0]
    assert not torch.equal(aa['raw_mask'], bb['raw_mask'])
    assert torch.equal(aa['chandra'], bb['chandra'])
    meta = a.get_resolution_metadata(0)
    assert meta['selected_radio_coverage'] == meta['coverage']['RAW']
    res = meta['target_resolution']
    physical = res['fwhm_arcsec'] * u.arcsec.to(u.rad) * Planck18.angular_diameter_distance(.2).to_value(u.kpc)
    assert res['fwhm_kpc'] == pytest.approx(physical)
    assert res['fwhm_over_r500'] == pytest.approx(physical/100)
    intersection = dataset(catalog)
    assert torch.equal(intersection[0]['raw_mask'].bool(), aa['raw_mask'].bool() & bb['raw_mask'].bool())
    # Changing mask policy reuses the physical pair.
    assert intersection.get_resolution_metadata(0)['cache_key'] == meta['cache_key']


def test_eligibility_is_explicit_pairwise_and_never_changes_keys(catalog):
    original = deepcopy(catalog[0])
    for source in ('RAW', 'T50kpcSUB'):
        ds = dataset(catalog, radio_source=source,
                     matched_resolution={'coverage_thresholds': {'CHANDRA': .99, 'intersection': .99}})
        assert ds.keys == [catalog[1]] and len(ds) == 1
        with pytest.raises(CoverageEligibilityError) as caught:
            ds[0]
        meta = ds.get_resolution_metadata(0)
        assert not meta['eligibility']['eligible']
        assert len(meta['eligibility']['reasons']) == 2
        assert caught.value.metadata['coverage'] == meta['coverage']
    assert catalog[0] == original
    report = dataset(catalog, matched_resolution={'coverage_thresholds': {'CHANDRA': .99}, 'coverage_policy': 'report'})
    assert not report[0]['resolution_metadata']['eligible']
    coverage = report.get_resolution_metadata(0)['coverage']
    boundary = dataset(catalog, matched_resolution={'coverage_thresholds': coverage})
    assert boundary[0]['resolution_metadata']['eligible']  # >= threshold


def test_cache_hit_no_reconvolution_and_corruption_recovery(catalog):
    ds = dataset(catalog)
    expected = ds[0]
    meta = ds.get_resolution_metadata(0)
    path = Path(meta['cache']['path'])
    with patch('clusterprep.matched_cache.prepare_matched_pair', side_effect=AssertionError('recomputed')):
        other = dataset(catalog, radio_source='T50kpcSUB')
        assert other.get_resolution_metadata(0)['cache']['status'] == 'reused'
        assert torch.equal(other[0]['chandra'], expected['chandra'])
    path.write_bytes(b'corrupt')
    assert ds.get_resolution_metadata(0)['cache']['status'] == 'rebuilt'
    assert torch.equal(ds[0]['raw'], expected['raw'])
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive['manifest'].item()))
        assert set(manifest['identity']['sources']) == {'RAW', 'T50kpcSUB', 'CHANDRA'}
        assert manifest['identity']['target_beam'] == meta['target_beam']
    assert not list(path.parent.glob('.building-*'))


@pytest.mark.parametrize('changed', ['RAW', 'T50kpcSUB', 'CHANDRA'])
def test_cache_invalidates_on_either_radio_or_chandra_even_in_memory(catalog, changed):
    ds = dataset(catalog, cache=True)
    before = ds.get_resolution_metadata(0)
    assert ds.get_resolution_metadata(0)['cache']['status'] == 'memory'
    info = catalog[0][catalog[1]]
    path = info['chandra_path'] if changed == 'CHANDRA' else info['radio_paths'][changed]
    with fits.open(path, mode='update') as hdul:
        if changed != 'CHANDRA':
            hdul[0].header['BMAJ'] *= 1.5
        else:
            hdul[0].data[128, 128] += 1.
    after = ds.get_resolution_metadata(0)
    assert before['cache_key'] != after['cache_key']
    assert len(list(catalog[2].rglob('*.npz'))) == 2
    if changed == 'CHANDRA':
        assert before['chandra_sha256'] != after['chandra_sha256']
    else:
        assert before['chandra_sha256'] == after['chandra_sha256']


def test_cache_parameters_geometry_versions_and_mutation_isolation(catalog):
    ds = dataset(catalog, cache=True)
    old = ds.get_resolution_metadata(0)
    changed = dataset(catalog, matched_resolution={'target_pixels_per_beam': 4., 'min_support': .99})
    assert old['cache_key'] != changed.get_resolution_metadata(0)['cache_key']
    smaller = dataset(catalog, size=64)
    assert old['cache_key'] != smaller.get_resolution_metadata(0)['cache_key']
    record = {**catalog[0][catalog[1]]['physical_record'], 'r500_kpc': 110}
    larger = dataset(catalog, matched_records={catalog[1]: record})
    assert old['cache_key'] != larger.get_resolution_metadata(0)['cache_key']
    ds._matched.store.implementation['pipeline_version'] += 1
    assert old['cache_key'] != ds.get_resolution_metadata(0)['cache_key']
    first = ds[0]
    first['raw'].zero_()
    first['chandra'].zero_()
    first['resolution_metadata']['coverage']['RAW'] = -1
    fresh = ds[0]
    assert fresh['raw'].any() and fresh['chandra'].any()
    assert fresh['resolution_metadata']['coverage']['RAW'] >= 0


@pytest.mark.parametrize('modalities', [None, ['radio', 'xray']])
def test_chandra_exact_with_rotation_flip_and_zoom(catalog, modalities):
    options = dict(cache=True, modalities=modalities, augment=True, training=True,
                   zoom={'enabled': True, 'factors': [1., .8, 1.25]})
    a = dataset(catalog, **options)
    b = dataset(catalog, radio_source='T50kpcSUB', **options)
    for index in (0, 1, 5, 25, 51):
        aa, bb = a[index], b[index]
        assert torch.equal(aa['chandra'], bb['chandra'])
        assert torch.equal(aa['chandra_mask'], bb['chandra_mask'])
        assert torch.equal(aa['raw_mask'], bb['raw_mask'])
        assert torch.all(aa['raw'][~aa['raw_mask'].bool()] == 0)


def test_no_disk_cache_and_optional_missing_chandra(catalog):
    info, cid, root = catalog
    info[cid]['chandra_path'] = None
    ds = dataset(catalog, mode='multimodal', processed_dir=None, cache=True)
    sample = ds[0]
    assert not sample['has_chandra'] and not sample['chandra'].any()
    assert not sample['chandra_mask'].any()
    meta = ds.get_resolution_metadata(0)
    assert meta['coverage']['CHANDRA'] == 0 and meta['cache']['path'] is None
    assert not root.exists()
    assert len(dataset(catalog, mode='pair')) == 0
    assert len(dataset(catalog, mode='multimodal', modalities=['radio', 'xray'], missing_modality_policy='required')) == 1
    ds = dataset(catalog, mode='raw', matched_resolution={'coverage_thresholds': {'CHANDRA': .1}})
    with pytest.raises(CoverageEligibilityError):
        ds[0]


def test_mass_derived_beam_and_train_normalization(catalog):
    info, cid, _ = catalog
    info[cid]['physical_record'].update(r500_kpc=None, m500_msun=1e13)
    ds = dataset(catalog, modalities=['radio', 'xray'],
                 normalization={'radio': {'scope': 'train'}, 'xray': {'scope': 'train'}})
    bounds = fit_normalization(ds)
    ds.normalization_bounds = bounds
    assert torch.isfinite(ds[0]['image']).all()
    meta = ds.get_resolution_metadata(0)
    assert meta['field']['fov_method'] == 'm500_derived'
    res = meta['target_resolution']
    assert res['fwhm_over_r500'] == pytest.approx(res['fwhm_kpc']/res['r500_kpc'])
    with pytest.raises(ValueError, match='Train normalization bounds missing'):
        dataset(catalog, modalities=['radio'], normalization={'radio': {'scope': 'train'}})[0]


@pytest.mark.parametrize('options', [{'radio_mask': 'union'}, {'coverage_policy': 'drop'},
    {'coverage_thresholds': {'RAW': 1.1}}, {'coverage_thresholds': {'bad': .5}},
    {'coverage_thresholds': {'CHANDRA': float('nan')}}, {'min_support': 0},
    {'target_pixels_per_beam': -1}, {'truncate': 2}, {'output_unit': 'K'}, {'typo': 1}])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        MatchedResolutionConfig.parse(options)


def test_unsupported_matched_combinations_do_not_fall_back(catalog):
    for kwargs in ({'scattering_dir': 'historical'}, {'radio_noise': {'enabled': True}},
                   {'modalities': ['radio', 'legacy']}, {'modalities': ['radio'], 'include_masks': False}):
        with pytest.raises(ValueError):
            dataset(catalog, **kwargs)
    del catalog[0][catalog[1]]['radio_paths']['T50kpcSUB']
    with pytest.raises(ValueError, match='T50kpcSUB'):
        dataset(catalog)


def test_historical_values_schema_and_cache_are_unchanged(catalog):
    info, cid, root = catalog
    original = deepcopy(info)
    kwargs = dict(mode='pair', task='rr_rh', size=128, processed_dir=root)
    implicit = ClusterDataset(info, [cid], ['RR', 'RH'], **kwargs)
    explicit = ClusterDataset(info, [cid], ['RR', 'RH'], radio_preprocessing='historical', **kwargs)
    a, b = implicit[0], explicit[0]
    linear = preprocess(info[cid], 'pair', 128)
    assert set(a) == {'raw', 'chandra', 'chandra_mask', 'cluster_id', 'key', 'has_chandra', 'label', 'label_mask'}
    for name in a:
        if isinstance(a[name], torch.Tensor):
            assert torch.equal(a[name], b[name])
    np.testing.assert_array_equal(a['raw'][0].numpy(), normalize(linear['raw']))
    np.testing.assert_array_equal(a['chandra'][0].numpy(), normalize(linear['chandra'], mask=linear['chandra_mask']))
    files = {p: p.read_bytes() for p in root.rglob('*.fits')}
    dataset(catalog)[0]
    assert all(p.read_bytes() == content for p, content in files.items())
    assert info == original and implicit.keys == explicit.keys == [cid]
    assert all('matched_resolution' not in str(p) for p in files)


def test_uncached_pair_and_cached_chandra_are_bitwise_identical(catalog):
    a = dataset(catalog, processed_dir=None)
    b = dataset(catalog, processed_dir=None, radio_source='T50kpcSUB',
                matched_resolution={'radio_mask': 'specific'})
    aa, bb = a[0], b[0]
    cached = dataset(catalog)[0]
    assert torch.equal(aa['chandra'], bb['chandra'])
    assert torch.equal(aa['chandra'], cached['chandra'])
    assert torch.equal(aa['chandra_mask'], bb['chandra_mask'])
    # X-ray training bounds also depend only on shared X-ray data and train keys.
    options = dict(modalities=['radio', 'xray'], normalization={'xray': {'scope': 'train'}})
    fit_a = fit_normalization(dataset(catalog, **options))
    fit_b = fit_normalization(dataset(catalog, radio_source='T50kpcSUB', **options))
    assert fit_a == fit_b


def test_source_change_during_computation_is_not_published(catalog):
    from clusterprep.matched_cache import prepare_matched_pair as original
    path = catalog[0][catalog[1]]['radio_paths']['T50kpcSUB']
    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        with fits.open(path, mode='update') as hdul:
            hdul[0].data[128, 128] += 1
        return result
    with patch('clusterprep.matched_cache.prepare_matched_pair', side_effect=changing):
        with pytest.raises(RuntimeError, match='Sources changed'):
            dataset(catalog)[0]
    assert not list(catalog[2].rglob('*.npz'))


def test_bad_record_fails_without_label_or_grid_fallback(catalog):
    info, cid, _ = catalog
    del info[cid]['physical_record']
    with pytest.raises(ValueError, match='matched_records'):
        dataset(catalog)
    with pytest.raises(ValueError, match='another cluster'):
        dataset(catalog, matched_records={cid: {'cluster_id': 'other'}})
    with pytest.raises(ValueError):
        dataset(catalog, matched_records={cid: {'ra_deg': 170, 'dec_deg': 30}})[0]


def test_required_modalities_preserve_generator_keys_and_cached_wcs_shape(catalog):
    info, cid, root = catalog
    ds = ClusterDataset(info, (key for key in [cid]), ['RR', 'RH'],
                        task='rr_rh', modalities=['radio', 'xray'],
                        missing_modality_policy='required', radio_preprocessing='matched_resolution',
                        processed_dir=root)
    assert ds.keys == [cid]
    fresh = ds._modality_images(ds.data_info[cid])
    cached = ds._modality_images(ds.data_info[cid])
    assert fresh['wcs'].array_shape == cached['wcs'].array_shape == (128, 128)
    assert fresh['wcs'].to_header() == cached['wcs'].to_header()
