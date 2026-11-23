"""Optical integration contracts, including real training/checkpoint lifecycle."""
from dataclasses import asdict, replace
from itertools import combinations
import json
import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.wcs import WCS
from torch.utils.data import DataLoader
from clusterprep.io import discover_legacy, load_optical_fits
from clusterprep.preprocessing import preprocess_optical, preprocess
from clusterprep.processed import load_processed_optical, load_processed, processed_paths
from clusterprep.dataset import ClusterDataset
from clusterprep.modalities import fit_normalization, channel_count
from clusterprep.training import ExperimentConfig, run_experiment, _inputs
from clusterprep.models import build_model
from clusterprep.modular_transfer import initialize_transfer, ModularTransferConfig, resolve_module


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def sources(tmp_path):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = [180, 45]
    wcs.wcs.crpix = [16, 16]
    wcs.wcs.cdelt = [-.01, .01]
    header = wcs.to_header()
    for i, band in enumerate('grz'):
        header[f'BAND{i}'] = band
    image = np.arange(1024, dtype=np.float32).reshape(32, 32)
    info = {}
    for i in range(12):
        cluster = f'c{i}'
        directory = tmp_path / cluster
        directory.mkdir()
        raw = directory / f'{cluster}.fits'
        fits.PrimaryHDU(image + i, wcs.to_header()).writeto(raw)
        if i != 3:
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.stack([image + i] * 3), header)]).writeto(
                directory / f'{cluster}_legacy.fits')
        info[str(i)] = dict(cluster_id=cluster, raw_path=str(raw), chandra_path=str(raw) if i != 2 else None,
                            labels={'NDE'} if i % 2 else {'DE', 'RR', 'RH'})
    return discover_legacy(info, tmp_path)


def test_read_projection_cache_and_invalidation(sources, tmp_path):
    info = sources['0']
    image, _, _ = load_optical_fits(info['legacy_path'], ['z', 'g'])
    assert image.shape == (2, 32, 32)
    optical = preprocess_optical(info, 8, ['g', 'r', 'z'])
    radio = preprocess(info, 'raw', 8)
    for band in optical['legacy']:
        np.testing.assert_allclose(band, radio['raw'], atol=1e-4)
    np.testing.assert_allclose(optical['wcs'].pixel_to_world_values(0, 0), radio['wcs'].pixel_to_world_values(0, 0))
    cache = tmp_path / 'cache'
    load_processed(info, 'pair', 8, cache)
    old = {k: p.stat().st_mtime_ns for k, p in processed_paths(info, 'pair', 8, cache).items()}
    assert load_processed_optical(info, 8, cache, return_status=True)[1] == 'created'
    assert load_processed_optical(info, 8, cache, return_status=True)[1] == 'reused'
    assert load_processed_optical(info, 8, cache, bands=['r'], return_status=True)[0]['legacy'].shape == (1, 8, 8)
    with fits.open(info['legacy_path'], mode='update') as hdus:
        hdus[1].data[:, :16] = np.nan
        hdus[1].data[:, 16, 16] = np.inf
    rebuilt, status = load_processed_optical(info, 8, cache, return_status=True)
    assert status == 'rebuilt'
    assert not rebuilt['legacy_mask'].all()
    assert old == {k: p.stat().st_mtime_ns for k, p in processed_paths(info, 'pair', 8, cache).items()}


def test_combinations_missing_and_geometry(sources, tmp_path):
    for n in range(1, 4):
        for selection in combinations(['radio', 'xray', 'legacy'], n):
            ds = ClusterDataset(sources, ['0', '2', '3'], ['DE', 'NDE'], size=8,
                                modalities=list(selection), processed_dir=tmp_path / 'cache')
            batch = next(iter(DataLoader(ds, batch_size=3)))
            assert batch['image'].shape == (3, channel_count(selection), 8, 8)
            assert torch.isfinite(batch['image']).all()
            if 'legacy' in selection:
                assert not batch['legacy_mask'][2].any()
                assert not batch['legacy'][2].any()
            if 'xray' in selection:
                assert not batch['chandra_mask'][1].any()
    ds = ClusterDataset(sources, ['0', '2', '3'], ['DE', 'NDE'], size=8,
                        modalities=['radio', 'xray', 'legacy'], missing_modality_policy='required',
                        augment=True, training=True, processed_dir=tmp_path / 'cache',
                        zoom={'enabled': True, 'factors': [1., 1.2]})
    assert ds.keys == ['0']
    for i in range(len(ds)):
        sample = ds[i]
        torch.testing.assert_close(sample['chandra'], sample['legacy'][:1])
        torch.testing.assert_close(sample['chandra_mask'], sample['legacy_mask'][:1])
        assert sample['cluster_id'] == 'c0'


def test_train_statistics_no_leakage(sources, tmp_path):
    options = {'bands': ['g'], 'normalization': {'scope': 'train'}}
    ds = ClusterDataset(sources, ['0'], ['DE', 'NDE'], size=8, modalities=['legacy'],
                        legacy=options, processed_dir=tmp_path / 'cache')
    before = fit_normalization(ds)
    with fits.open(sources['1']['legacy_path'], mode='update') as hdus:
        hdus[1].data[:] *= 10000
    assert fit_normalization(ds) == before
    with pytest.raises(ValueError, match='bounds missing'):
        ds[0]
    ds.normalization_bounds = before
    assert torch.isfinite(ds[0]['image']).all()


@pytest.mark.parametrize('model_name', ['cnn', 'image_cnn', 'multi_encoder_cnn'])
def test_training_reload_evaluation(sources, tmp_path, model_name):
    cfg = ExperimentConfig(name=model_name, model=model_name, mode='multimodal', task='rr_rh',
        modalities=['radio', 'xray', 'legacy'], legacy={'bands': ['g'], 'normalization': {'scope': 'train'}},
        size=8, width=2, device='cpu', batch_size=2, epochs=1, augment=False,
        processed_dir=str(tmp_path / 'cache'), verbose=False)
    splits = ([str(i) for i in range(4)], [str(i) for i in range(4, 8)], [str(i) for i in range(8, 12)])
    run = run_experiment(sources, cfg, tmp_path / 'runs', splits=splits)
    assert json.loads((run / 'status.json').read_text())['status'] == 'completed'
    metadata = json.loads((run / 'config.json').read_text())
    restored = ExperimentConfig(**metadata['config'])
    model = build_model(restored, 2).eval()
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    ds = ClusterDataset(sources, splits[2], ['RR', 'RH'], task='rr_rh', size=8,
        modalities=cfg.modalities, legacy=cfg.legacy, normalization_bounds=restored.normalization_bounds,
        processed_dir=tmp_path / 'cache')
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert model(_inputs(batch, cfg.mode)).shape == (2, 2)
    saved_splits = json.loads((run / 'splits.json').read_text())
    assert saved_splits['train'] == splits[0]


def test_transfer_keeps_optical_fresh(sources, tmp_path):
    source_config = ExperimentConfig(name='source', model='dual_encoder_cnn', mode='multimodal', size=8)
    source = build_model(source_config, 2)
    path = tmp_path / 'source.pt'
    torch.save({'config': asdict(source_config), 'class_names': ['DE', 'NDE'], 'model_state_dict': source.state_dict()}, path)
    target_config = replace(source_config, name='target', model='multi_encoder_cnn',
                            modalities=['radio', 'xray', 'legacy'])
    target = build_model(target_config, 2)
    before = {k: v.clone() for k, v in target.encoders['legacy'].state_dict().items()}
    transfer = ModularTransferConfig(enabled=True, sources={'old': {'checkpoint': str(path)}},
        mappings=[{'source': 'old', 'from': name, 'to': name} for name in ['radio', 'xray']])
    initialize_transfer(target, transfer)
    for name in ['radio', 'xray']:
        for k, value in resolve_module(source, name).state_dict().items():
            torch.testing.assert_close(value, resolve_module(target, name).state_dict()[k])
    for k, value in before.items():
        torch.testing.assert_close(value, target.encoders['legacy'].state_dict()[k])


def test_no_coverage_and_invalid_dimensions(sources, tmp_path):
    info = sources['0']
    with fits.open(info['legacy_path'], mode='update') as hdus:
        hdus[1].header['CRVAL1'] = 0
        hdus[1].header['CRVAL2'] = -45
    image = preprocess_optical(info, 8, ['g'])
    assert not image['legacy_mask'].any()
    with fits.open(info['legacy_path'], mode='update') as hdus:
        hdus[1].data = np.zeros((2, 3, 32, 32))
    with pytest.raises(ValueError, match='Expected'):
        load_optical_fits(info['legacy_path'])


def test_required_exclusions_preserve_split_membership(sources, tmp_path):
    cfg = ExperimentConfig(name='required', model='cnn', task='rr_rh', modalities=['radio', 'xray', 'legacy'],
        missing_modality_policy='required', size=8, device='cpu', epochs=1, augment=False,
        batch_size=2, processed_dir=str(tmp_path / 'cache'), verbose=False)
    splits = (['0', '1', '2', '3'], ['4', '5'], ['6', '7'])
    run = run_experiment(sources, cfg, tmp_path / 'runs', splits=splits)
    manifest = json.loads((run / 'splits.json').read_text())
    assert manifest['base_splits']['train'] == splits[0]
    assert manifest['train'] == ['0', '1']
    assert manifest['test'] == splits[2]
    assert json.loads((run / 'modality_exclusions.json').read_text())['train'] == ['2', '3']


def test_new_json_examples():
    from pathlib import Path
    for name, channels in [('legacy_multichannel', 9), ('legacy_multimodal', 9)]:
        spec = json.loads((Path(__file__).parents[1] / 'examples' / (name + '.json')).read_text())
        cfg = ExperimentConfig(**spec['experiments'][0])
        assert channel_count(cfg.modalities, cfg.legacy, cfg.include_masks) == channels
        model = build_model(replace(cfg, size=8), 2).eval()
        assert model(torch.zeros(2, channels, 8, 8)).shape == (2, 2)


def test_preprocess_cli_optical_only(sources, tmp_path, capsys):
    import shutil
    from clusterprep.__main__ import preprocessing_main
    root = tmp_path / 'classified'
    (root / 'RAW' / 'DE').mkdir(parents=True)
    (root / 'RAW' / 'NDE').mkdir(parents=True)
    shutil.copyfile(sources['0']['raw_path'], root / 'RAW' / 'DE' / 'c0.fits')
    cache = tmp_path / 'cli-cache'
    args = [str(root), '--legacy-root', str(tmp_path), '--legacy-only', '--size', '8', '--processed-dir', str(cache)]
    preprocessing_main(args)
    assert json.loads(capsys.readouterr().out)['created'] == 1
    preprocessing_main(args)
    assert json.loads(capsys.readouterr().out)['reused'] == 1
    assert not list(cache.glob('*/*RAW*'))
    assert not list(cache.glob('*/*CHANDRA*'))


def test_cv_required_preserves_folds_and_fits_train_only(sources, tmp_path):
    from clusterprep.cross_validation import build_internal_folds
    cfg = ExperimentConfig(name='cv', model='cnn', task='rr_rh', modalities=['radio', 'xray', 'legacy'],
        missing_modality_policy='required', legacy={'bands': ['g'], 'normalization': {'scope': 'train'}},
        cross_validation={'enabled': True, 'n_splits': 2, 'max_epochs': 1, 'retrain_full_train': False},
        size=8, device='cpu', epochs=1, augment=False, batch_size=2,
        processed_dir=str(tmp_path / 'cache'), verbose=False)
    splits = (['0', '1', '2', '3', '4', '5'], ['6', '7', '8', '9'], ['10', '11'])
    expected = build_internal_folds(sources, splits[0] + splits[1], cfg)
    run = run_experiment(sources, cfg, tmp_path / 'runs', splits=splits)
    assert json.loads((run / 'cv' / 'folds.json').read_text()) == expected
    oof = json.loads((run / 'cv' / 'oof_predictions.json').read_text())
    assert {r['key'] for r in oof} == set(splits[0] + splits[1]) - {'2', '3'}
    for fold in expected['folds']:
        record = json.loads((run / 'cv' / 'trial_0' / f"fold_{fold['fold']}" / 'normalization.json').read_text())
        assert set(record['train_keys']) == set(fold['train']) - {'2', '3'}


def test_explicit_first_convolution_reinitialization(tmp_path):
    cfg = ExperimentConfig(name='source', model='image_cnn', size=8)
    source = build_model(cfg, 2)
    path = tmp_path / 'source.pt'
    torch.save({'config': asdict(cfg), 'class_names': ['DE', 'NDE'], 'model_state_dict': source.state_dict()}, path)
    target = build_model(replace(cfg, modalities=['radio', 'legacy']), 2)
    before = target.encoder.encoder[0].weight.detach().clone()
    transfer = ModularTransferConfig(enabled=True, sources={'old': {'checkpoint': str(path)}}, mappings=[
        {'source': 'old', 'from': 'radio.encoder.0', 'to': 'radio.encoder.0', 'on_incompatible': 'skip'},
        {'source': 'old', 'from': 'radio.to_latent', 'to': 'radio.to_latent', 'on_incompatible': 'error'}])
    report = initialize_transfer(target, transfer)
    assert [m['status'] for m in report['mappings']] == ['skipped', 'transferred']
    torch.testing.assert_close(before, target.encoder.encoder[0].weight)
    for k, v in source.encoder.to_latent.state_dict().items():
        torch.testing.assert_close(v, target.encoder.to_latent.state_dict()[k])


def test_missing_encoder_gating_and_plot(sources, tmp_path):
    import matplotlib.pyplot as plt
    from clusterprep.visualization import plot_sample
    ds = ClusterDataset(sources, ['3'], ['DE', 'NDE'], modalities=['radio', 'xray', 'legacy'],
                        size=8, processed_dir=tmp_path / 'cache')
    sample = ds[0]
    model = build_model(ExperimentConfig(name='gating', model='multi_encoder_cnn', size=8,
                                        modalities=['radio', 'xray', 'legacy']), 2).eval()
    model(sample['image'][None]).sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in model.encoders['legacy'].parameters())
    figure = plot_sample(sample, tmp_path / 'plot.png', chandra_contours=True)
    assert any(t.get_text() == 'Observation absente' for ax in figure.axes for t in ax.texts)
    plt.close(figure)
