"""Selected LOFAR product: inventory, physical grids, provenance and training."""
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.wcs import WCS

from clusterprep import ClusterDataset, ExperimentConfig, run_experiment, compare_experiments
from clusterprep.io import get_data_info_dict, resolve_radio_path, select_radio_source, radio_cluster_id
from clusterprep.preprocessing import preprocess
from clusterprep.processed import _identity, load_processed, processed_paths
from clusterprep.scattering import scattering_identity, scattering_paths


def write_image(path, side=16, scale=.01, seed=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = [180, 45]
    wcs.wcs.crpix = [side / 2 + .5, side / 2 + .5]
    wcs.wcs.cdelt = [-scale, scale]
    header = wcs.to_header()
    header.update(BMAJ=scale*3, BMIN=scale*2, BPA=35., BUNIT='JY/BEAM')
    data = np.random.default_rng(seed).normal(size=(side, side)).astype('float32')
    fits.PrimaryHDU(data, header).writeto(path, overwrite=True)
    return path


@pytest.fixture
def catalog(tmp_path):
    root = tmp_path / 'classified'
    for i in range(12):
        label = 'DE' if i % 2 == 0 else 'NDE'
        cid = f'C{i:02}'
        write_image(root/'RAW'/label/f'{cid}.fits', seed=i)
        tapered = write_image(root/'T25kpc'/label/f'{cid}T25kpc.fits', side=8, scale=.02, seed=i+12)
        write_image(root/'T50kpc'/label/f'{cid}_T50kpc.fits', side=8, scale=.04, seed=i+24)
        dest = root/'CHANDRA'/label/f'{cid}CHANDRA.fits'
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tapered, dest)
    return root, get_data_info_dict(root)


def test_inventory_variants_duplicates_orphans_and_labels(catalog):
    root, _ = catalog
    for product, filename in [('RAW','C00.fits'), ('T25kpc','C00T25kpc.fits')]:
        dest = root/product/'RH'/filename
        dest.parent.mkdir(parents=True)
        shutil.copyfile(root/product/'DE'/filename, dest)
    write_image(root/'T25kpc'/'RH'/'orphanT25kpc.fits')
    write_image(root/'T25kpc'/'RH'/'unrecognised.fits')
    info, audit = get_data_info_dict(root, return_audit=True)
    assert info['C00.fits']['labels'] == {'DE','RH'}
    assert len(audit['T25kpc']['duplicates']) == 1
    assert len(audit['T25kpc']['orphans']) == 2
    assert len(audit['T25kpc']['naming_incompatibilities']) == 1
    for source in ['RAW', 'T25kpc', 'T50kpc']:
        assert Path(resolve_radio_path(info['C00.fits'], source)).parent.parent.name == source
    write_image(root/'T25kpc'/'RH'/'C00T25kpc.fits', seed=99)
    with pytest.raises(ValueError, match='contenus différents'):
        get_data_info_dict(root)
    with pytest.raises(ValueError, match='ambiguë'):
        radio_cluster_id('AT25kpc.fits', 'T25kpc', {'A', 'AT25kpc'})


def test_strict_missing_no_fallback_and_config_validation(catalog):
    _, info = catalog
    del info['C00.fits']['radio_paths']['T25kpc']
    with pytest.raises(ValueError, match='T25kpc.*C00'):
        select_radio_source(info, 'T25kpc')
    with pytest.raises(ValueError, match='inconnu'):
        select_radio_source(info, 'T999kpc')
    for source in ['../T25kpc', '/tmp/a', 'a/b', 'a\\b', '']:
        with pytest.raises(ValueError):
            ExperimentConfig('bad', radio_source=source)
    with pytest.raises(ValueError):
        ExperimentConfig('bad', missing_radio_policy='drop')


def test_raw_compatibility_and_cache_isolation(catalog, tmp_path):
    _, info = catalog
    old = {k: v for k,v in info['C00.fits'].items() if k not in {'radio_paths','available_radio_sources'}}
    raw = select_radio_source(info)['C00.fits']
    tapered = select_radio_source(info, 'T25kpc')['C00.fits']
    assert raw['raw_path'] == tapered['raw_path'] == old['raw_path']
    assert _identity(raw, 'pair', 8) == _identity(old, 'pair', 8)
    assert processed_paths(raw, size=8) == processed_paths(old, size=8)
    cache = tmp_path/'processed'
    a = load_processed(old, 'pair', 8, cache)
    _, status = load_processed(raw, 'pair', 8, cache, return_status=True)
    assert status == 'reused'
    b, status = load_processed(tapered, 'pair', 8, cache, return_status=True)
    assert status == 'created'
    assert not np.allclose(a['raw'], b['raw'])
    assert processed_paths(raw, size=8, processed_dir=cache) != processed_paths(tapered, size=8, processed_dir=cache)
    identity = _identity(tapered, 'raw', 8)
    write_image(Path(resolve_radio_path(tapered)), side=8, scale=.02, seed=87)
    assert _identity(tapered, 'raw', 8) != identity
    assert load_processed(tapered, 'pair', 8, cache, return_status=True)[1] == 'rebuilt'
    assert scattering_identity(raw, 'raw', 8, 1, 2, 1) != scattering_identity(tapered, 'raw', 8, 1, 2, 1)
    assert scattering_paths(raw, 'raw', 8, 1, 2, 1, cache) != scattering_paths(tapered, 'raw', 8, 1, 2, 1, cache)


@pytest.mark.parametrize('size', [4,8])
def test_selected_wcs_alignment_mask_and_beam(catalog, size):
    _, info = catalog
    item = select_radio_source(info, 'T25kpc')['C00.fits']
    result = preprocess(item, 'pair', size)
    np.testing.assert_allclose(result['raw'], result['chandra'], atol=1e-5)
    assert result['chandra_mask'].all()
    assert result['radio_metadata']['BMAJ'] == pytest.approx(.06)
    assert abs(result['wcs'].pixel_scale_matrix[0,0]) == pytest.approx(.02 * (8//size))
    with pytest.raises(ValueError, match='trop petite'):
        preprocess(item, 'pair', 9)


@pytest.mark.parametrize('mode,modalities', [('raw',None),('pair',None),('multimodal',None),('raw',['radio','xray'])])
def test_dataset_channels_and_aligned_augmentation(catalog, mode, modalities):
    _, info = catalog
    kwargs = dict(mode=mode, size=8, processed_dir=None, modalities=modalities, augment=True, training=True)
    raw = ClusterDataset(info, ['C00.fits'], ['DE','NDE'], **kwargs)
    tapered = ClusterDataset(info, ['C00.fits'], ['DE','NDE'], radio_source='T25kpc', **kwargs)
    a,b = raw[1],tapered[1]
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], torch.Tensor):
            assert a[key].shape == b[key].shape
    if modalities is None and mode != 'raw':
        torch.testing.assert_close(b['raw'], b['chandra'], atol=1e-5, rtol=1e-5)
    assert tapered.get_item_info(0)['radio_source'] == 'T25kpc'


def test_one_epoch_inference_common_split_transfer_and_cv(catalog, tmp_path):
    from clusterprep.cross_validation import build_internal_folds
    from clusterprep.experiment_loader import resolve_transfer_splits
    from clusterprep.models import build_model
    from clusterprep.modular_transfer import ModularTransferConfig, initialize_transfer
    _, info = catalog
    splits = (list(info)[:8],list(info)[8:10],list(info)[10:])
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = ExperimentConfig('radio', size=8, width=2, batch_size=2, epochs=1,
            device='cpu', use_compile=False, augment=False, use_scheduler=False,
            processed_dir=str(tmp_path/'cache'), verbose=False)
        runs = [run_experiment(info, replace(cfg, radio_source=s), tmp_path/'runs', splits=splits)
                for s in ['RAW','T25kpc']]
        rows = compare_experiments(runs, cohort='all')
        assert {r['radio_source'] for r in rows} == {'RAW','T25kpc'}
        assert len({r['cohort_id'] for r in rows}) == 1
        for run, source in zip(runs, ['RAW','T25kpc']):
            meta = json.loads((run/'splits.json').read_text())
            assert meta['radio_source'] == source
            assert meta['train'] == splits[0]
            assert (run/'test_predictions.csv').exists()
            assert json.loads((run/'metrics.json').read_text())['epochs_ran'] == 1
        selected = select_radio_source(info,'T25kpc')
        resolved, _ = resolve_transfer_splits(runs[0], selected, task='de_nde', mode='raw')
        assert tuple(resolved) == splits
        cv = replace(cfg, cross_validation={'enabled':True,'n_splits':2})
        assert build_internal_folds(info,splits[0],cv) == build_internal_folds(selected,splits[0],replace(cv,radio_source='T25kpc'))
        target = build_model(replace(cfg, radio_source='T25kpc'), 2)
        plan = ModularTransferConfig(enabled=True, sources={'raw':{'checkpoint':str(runs[0]/'best.pt')}},
            mappings=[{'source':'raw','from':'radio','to':'radio'}])
        report = initialize_transfer(target,plan)
        assert report['mappings'][0]['status'] == 'transferred'
        source_model = build_model(cfg,2)
        source_model.load_state_dict(torch.load(runs[0]/'best.pt',weights_only=True)['model_state_dict'])
        from clusterprep.modular_transfer import resolve_module
        for k,v in resolve_module(source_model,'radio').state_dict().items():
            torch.testing.assert_close(v,resolve_module(target,'radio').state_dict()[k])
    finally:
        torch.set_num_threads(old_threads)


def test_gallery_uses_selected_cache(catalog, tmp_path):
    from clusterprep.multilabel_analysis import _prepare_cluster_modalities
    _, info = catalog
    item = select_radio_source(info,'T25kpc')['C00.fits']
    expected = load_processed(item,'pair',8,tmp_path/'cache')
    entry = {**item, 'key':'C00.fits', 'size':8, 'has_chandra':True}
    radio,chandra = _prepare_cluster_modalities(entry,processed_root=tmp_path/'cache',
        radio_template='wrong_RAW.fits',chandra_template='wrong_CHANDRA.fits')
    np.testing.assert_array_equal(radio,expected['raw'])
    np.testing.assert_array_equal(chandra,expected['chandra'])


def test_train_normalization_is_source_specific_and_ignores_test_pixels(catalog):
    from clusterprep.modalities import fit_normalization
    _, info = catalog
    keys = list(info)[:4]
    def fit(source):
        ds = ClusterDataset(info,keys,['DE','NDE'],size=8,processed_dir=None,
            radio_source=source,modalities=['radio'],normalization={'radio':{'scope':'train'}})
        return fit_normalization(ds)
    raw,tapered = fit('RAW'),fit('T25kpc')
    assert raw != tapered
    write_image(Path(info['C11.fits']['radio_paths']['T25kpc']),side=8,seed=999)
    assert fit('T25kpc') == tapered


def test_optical_reference_and_cache_follow_radio(catalog,tmp_path):
    from clusterprep.processed import load_processed_optical
    _,info = catalog
    selected = select_radio_source(info,'T25kpc')['C00.fits']
    selected['legacy_path'] = resolve_radio_path(selected)
    result = load_processed_optical(selected,size=8,processed_dir=tmp_path/'cache')
    np.testing.assert_allclose(result['legacy'][0],preprocess(selected,'raw',8)['raw'],atol=1e-5)
    assert result['legacy_mask'].all()
    assert list((tmp_path/'cache'/'T25kpc').rglob('*LEGACY_processed*.fits'))


def test_tapered_cross_validation_runs_on_selected_source(catalog,tmp_path):
    _,info = catalog
    keys=list(info)
    cfg=ExperimentConfig('tapered_cv',radio_source='T25kpc',size=8,width=2,batch_size=2,
        epochs=1,device='cpu',use_compile=False,augment=False,use_scheduler=False,
        processed_dir=None,verbose=False,cross_validation={'enabled':True,'n_splits':2,'max_epochs':1})
    old_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        root=run_experiment(info,cfg,tmp_path/'runs',splits=(keys[:6],keys[6:8],keys[8:]))
    finally:
        torch.set_num_threads(old_threads)
    manifests=list(root.parent.rglob('splits.json'))
    assert len(manifests)==3
    for path in manifests:
        manifest=json.loads(path.read_text())
        assert manifest['radio_source']=='T25kpc'
        assert not set(manifest['train']) & set(manifest['validation'])
        assert all('/T25kpc/' in v['active_radio_path'] for v in manifest['samples'].values())
        if path.parent.name != 'final':
            assert not set(manifest['train']+manifest['validation']) & set(keys[8:])
    # The experiment view records absolute paths for inference from another cwd.
    selected=select_radio_source(info,'T25kpc')
    assert all(Path(resolve_radio_path(v)).is_absolute() for v in selected.values())
