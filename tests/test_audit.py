"""Régressions de l'audit, exclusivement sur FITS synthétiques."""
import csv
from contextlib import nullcontext
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.wcs import WCS
from torch.utils.data import DataLoader

from clusterprep import ClusterDataset, ExperimentConfig, build_model, get_data_info_dict
from clusterprep.labels import task_target
from clusterprep.metrics import multilabel_metrics
from clusterprep.scattering import prepare_scattering, load_scattering, scattering_paths
from clusterprep.splits import split_data_keys
from clusterprep.training import Trainer, run_experiment, run_experiments, compare_experiments
from clusterprep.training_core import build_criterion


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def catalog(tmp_path):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [8, 8]
    w.wcs.crval = [180, 45]
    w.wcs.cdelt = [-.01, .01]
    rng = np.random.default_rng(12)
    combinations = [{'RR'}, {'RH'}, {'RR', 'RH'}, {'cRR'}, {'cRH'}, {'cRR', 'cRH'}]
    for i in range(36):
        labels = {'NDE'} if i % 6 == 5 else combinations[i % 6] | {'DE'}
        image = rng.normal(size=(16, 16)).astype('float32')
        image[0, 0] = np.nan
        for label in labels:
            directory = tmp_path / 'RAW' / label
            directory.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(image, w.to_header()).writeto(directory / f'c{i}.fits')
        if i % 3:
            directory = tmp_path / 'CHANDRA' / 'RH'
            directory.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(image, w.to_header()).writeto(directory / f'c{i}CHANDRA.fits')
    return get_data_info_dict(tmp_path)


def test_inventory_and_conflicts(catalog, tmp_path):
    assert len(catalog) == 36
    assert catalog['c2.fits']['labels'] == {'DE', 'RR', 'RH'}
    assert catalog['c1.fits']['chandra_path']
    p = tmp_path / 'RAW' / 'RH' / 'c2.fits'
    with fits.open(p, mode='update') as hdus:
        hdus[0].data[1, 1] += 1
    with pytest.raises(ValueError, match='différent'):
        get_data_info_dict(tmp_path)


def test_chandra_conflict(catalog, tmp_path):
    p = tmp_path / 'CHANDRA' / 'RR'
    p.mkdir()
    data, header = fits.getdata(catalog['c1.fits']['chandra_path'], header=True)
    fits.PrimaryHDU(data + 1, header).writeto(p / 'c1CHANDRA.fits')
    with pytest.raises(ValueError, match='CHANDRA contradictoires'):
        get_data_info_dict(tmp_path)


@pytest.mark.parametrize('mode', ['raw', 'pair', 'multimodal'])
@pytest.mark.parametrize('task', ['de_nde', 'rr_rh'])
def test_dataset_modes_targets_variants(catalog, tmp_path, mode, task):
    ds = ClusterDataset(catalog, ['c2.fits'], ['DE', 'NDE'], task=task,
                        mode=mode, size=16, augment=True, cache=True,
                        processed_dir=tmp_path / 'cache')
    assert len(ds) == 24
    for i in range(24):
        sample = ds[i]
        assert sample['label'].dtype == (torch.long if task == 'de_nde' else torch.float32)
        assert sample['label'].tolist() == (0 if task == 'de_nde' else [1, 1])
        assert torch.isfinite(sample['raw']).all()
        if mode != 'raw':
            from torchvision.transforms import functional as TF, InterpolationMode
            base = ClusterDataset(catalog, ['c2.fits'], ['DE', 'NDE'], task=task,
                                  mode=mode, size=16, processed_dir=None)[0]
            for channel in ['raw', 'chandra', 'chandra_mask']:
                expected = TF.hflip(base[channel]) if i % 2 else base[channel]
                expected = TF.rotate(expected, 30 * (i // 2), interpolation=InterpolationMode.NEAREST)
                torch.testing.assert_close(sample[channel], expected)
            assert torch.isfinite(sample['chandra']).all()
            assert set(sample['chandra_mask'].unique().tolist()) <= {0., 1.}
    ds[0]['raw'].zero_()
    assert ds[0]['raw'].sum() > 0


def test_candidates_and_legacy(catalog):
    for policy, expected, mask in [('positive', [1, 0], [1, 1]), ('ignore', [0, 0], [0, 1])]:
        t, m = task_target(catalog['c3.fits'], 'rr_rh', policy)
        assert t.tolist() == expected and m.tolist() == mask
    assert task_target({'labels': {'RR', 'cRR'}}, 'rr_rh', 'ignore')[1].tolist() == [1, 1]
    assert task_target({'class': 'NDE'}, 'de_nde')[0] == 1
    for invalid in ['all', 'typo']:
        with pytest.raises(ValueError):
            ClusterDataset(catalog, [], ['DE', 'NDE'], candidate_policy=invalid)
    with pytest.raises(ValueError, match='ambigu'):
        task_target({'labels': {'DE', 'NDE'}}, 'de_nde')


def test_missing_chandra_and_memory_invalidation(catalog):
    ds = ClusterDataset(catalog, ['c0.fits'], ['DE', 'NDE'], task='rr_rh',
                        mode='multimodal', size=16, cache=True, processed_dir=None)
    original = ds[0]['raw'].clone()
    assert ds[0]['chandra'].sum() == ds[0]['chandra_mask'].sum() == 0
    with fits.open(catalog['c0.fits']['raw_path'], mode='update') as hdus:
        hdus[0].data[:] = 0
    assert not torch.equal(original, ds[0]['raw'])
    with pytest.raises(ValueError, match='CHANDRA'):
        ClusterDataset(catalog, ['c0.fits'], ['DE', 'NDE'], mode='pair')


@pytest.mark.parametrize('task', ['de_nde', 'rr_rh'])
def test_cluster_splits_and_rare_combinations(catalog, task):
    for folds in [None, 3]:
        options = dict(task=task, n_folds=folds, seed=9)
        splits = split_data_keys(catalog, **options)
        assert splits == split_data_keys(catalog, **options)
        ids = [{catalog[k]['cluster_id'] for k in keys} for keys in splits]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        assert len(set.union(*ids)) == len(catalog)
    rare = dict(list(catalog.items())[:9])
    with pytest.warns(UserWarning, match='Stratification'):
        split_data_keys(rare, task='rr_rh')


@pytest.mark.parametrize('mode', ['raw', 'multimodal'])
def test_scattering_equivalence_cache_and_labels(catalog, tmp_path, mode):
    Scattering2D = pytest.importorskip('kymatio.torch').Scattering2D
    options = dict(mode=mode, size=16, J=2, L=2, max_order=2,
                   processed_dir=None, scattering_dir=tmp_path / 'scat', verbose=False)
    info = catalog['c2.fits']
    # Precomputation must not even require labels.
    unlabeled = {'k': {k: v for k, v in info.items() if k != 'labels'}}
    assert prepare_scattering(unlabeled, ['k', 'k'], [], **options)['created'] == 1
    params = {k: options[k] for k in ['mode', 'size', 'J', 'L', 'max_order', 'scattering_dir']}
    coeff = load_scattering(info, **params)
    ds = ClusterDataset(catalog, ['c2.fits'], ['DE', 'NDE'], task='rr_rh',
                        mode=mode, size=16, augment=True, processed_dir=None)
    direct = Scattering2D(J=2, L=2, shape=(16, 16))
    for i in range(24):
        sample = ds[i]
        names = ['raw'] if mode == 'raw' else ['raw', 'chandra', 'chandra_mask']
        x = torch.cat([sample[n] for n in names])[None]
        np.testing.assert_allclose(coeff[i], direct(x)[0].numpy(), atol=2e-6, rtol=2e-5)
    assert prepare_scattering(catalog, ['c2.fits'], [], task='rr_rh', **options)['reused'] == 1
    paths = scattering_paths(info, **params)
    before = paths['data'].stat().st_mtime_ns
    catalog['c2.fits']['labels'] = {'NDE'}
    assert prepare_scattering(catalog, ['c2.fits'], [], **options)['reused'] == 1
    assert paths['data'].stat().st_mtime_ns == before
    damaged = np.load(paths['data'])
    damaged[0, 0, 0, 0, 0] += 1
    np.save(paths['data'], damaged)
    with pytest.raises(ValueError, match='Checksum'):
        load_scattering(info, **params)
    assert prepare_scattering(catalog, ['c2.fits'], [], **options)['created'] == 1
    with fits.open(info['raw_path'], mode='update') as h:
        h[0].data[1, 1] += 1
    with pytest.raises(ValueError, match='invalide'):
        load_scattering(info, **params)
    assert prepare_scattering(catalog, ['c2.fits'], [], **options)['created'] == 1
    with pytest.raises(FileNotFoundError):
        load_scattering(info, **{**params, 'J': 1})


@pytest.mark.parametrize('target', [[[0, 0]] * 3, [[1, 1]] * 3, [[1, 0], [0, 1], [1, 1]]])
def test_metrics_degenerate_and_masks(target):
    probabilities = np.zeros_like(target, dtype=float)
    result = multilabel_metrics(target, probabilities)
    json.dumps(result, allow_nan=False)
    assert result['micro_f1'] == 0
    masked = multilabel_metrics(target, probabilities, mask=np.zeros_like(target))
    assert masked['macro_f1'] is None and masked['subset_accuracy'] is None
    json.dumps(masked, allow_nan=False)
    with pytest.raises(ValueError):
        multilabel_metrics([[.5, 1]], [[.5, .5]])


def test_masked_loss_gradient_weights_and_metrics():
    config = ExperimentConfig('masked', task='rr_rh', candidate_policy='ignore', device='cpu',
                              use_scheduler=False, use_mixup=False, use_class_weights=False)
    model = torch.nn.Linear(2, 2, bias=False)
    data = [dict(raw=torch.tensor([1., 0.]), label=torch.tensor([0., 1.]), label_mask=torch.tensor([0., 1.])),
            dict(raw=torch.tensor([0., 1.]), label=torch.tensor([1., 0.]), label_mask=torch.tensor([1., 1.]))]
    trainer = Trainer(model, config, ['RR', 'RH'], [1, 1], n_train=[1, 2],
                      input_adapter=lambda b: b['raw'], steps_per_epoch=1)
    metrics, _ = trainer.evaluate(DataLoader(data, batch_size=1))
    logits = model(torch.eye(2))
    target, mask = torch.stack([d['label'] for d in data]), torch.stack([d['label_mask'] for d in data])
    expected = (torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none') * mask).sum() / mask.sum()
    assert metrics['loss'] == pytest.approx(expected.item())
    assert metrics['per_label']['RR']['n_known'] == 1
    trainer.train_one_epoch(DataLoader(data, batch_size=2))
    assert model.weight.grad[0, 0] == 0
    criterion = build_criterion(replace(config, use_class_weights=True), [1, 2], n_train=[3, 3])
    torch.testing.assert_close(criterion.pos_weight, torch.tensor([2., .5]))
    assert isinstance(build_criterion(replace(config, task='de_nde'), [1, 2]), torch.nn.CrossEntropyLoss)
    proba = trainer.predict(DataLoader(data, batch_size=2))
    assert proba.shape == (2, 2)


@pytest.mark.parametrize('model,mode', [('cnn','raw'), ('fusion','multimodal'), ('image_cnn','raw'),
                                     ('dual_encoder_cnn','pair'), ('dual_ssn','raw'), ('dual_encoder_ssn','multimodal')])
def test_model_logits_finite_gradients_no_online_scattering(model, mode):
    config = ExperimentConfig('model', model=model, mode=mode, size=16,
                              model_params={'L': 2, 'hidden_dim2': 2} if 'ssn' in model else {})
    net = build_model(config, 2)
    x = torch.rand(2, 1 if mode == 'raw' else 3, 16, 16)
    if mode != 'raw':
        x[:, 2] = 0
    args = [x]
    if 'ssn' in model:
        Scattering2D = pytest.importorskip('kymatio.torch').Scattering2D
        args.append(Scattering2D(J=2, L=2, shape=(16,16))(x))
    guard = patch('kymatio.torch.Scattering2D.forward', side_effect=AssertionError('online scattering')) if 'ssn' in model else nullcontext()
    with guard:
        out = net(*args)
        assert out.shape == (2, 2) and torch.isfinite(out).all()
        torch.nn.functional.binary_cross_entropy_with_logits(out, torch.ones_like(out)).backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        net.eval()
        assert torch.isfinite(net(*args)).all()


@pytest.mark.parametrize('task,model', [('de_nde','cnn'), ('rr_rh','cnn'), ('rr_rh','dual_ssn'), ('rr_rh','dual_encoder_ssn')])
def test_two_epoch_integration(catalog, tmp_path, task, model):
    if 'ssn' in model:
        pytest.importorskip('kymatio')
    config = ExperimentConfig('integration', task=task, model=model, mode='multimodal', size=16,
        epochs=2, batch_size=4, width=2, device='cpu', augment=False, use_mixup=True,
        processed_dir=str(tmp_path/'cache'), verbose=False,
        model_params={'L': 2, 'hidden_dim2': 2} if 'ssn' in model else {})
    splits = ([f'c{i}.fits' for i in range(12)], [f'c{i}.fits' for i in range(12,18)],
              [f'c{i}.fits' for i in range(18,24)])
    run = run_experiment(catalog, config, tmp_path/'runs', splits=splits)
    for name in ['best.pt', 'history.csv', 'history.jsonl', 'config.json', 'splits.json',
                 'validation_predictions.csv', 'test_predictions.csv', 'metrics.json', 'split_distribution.json']:
        assert (run/name).stat().st_size > 0
    distribution = json.loads((run/'split_distribution.json').read_text())
    assert [sum(distribution[name].values()) for name in ['train','validation','test']] == [12,6,6]
    result = json.loads((run/'metrics.json').read_text())
    assert result['epochs_ran'] == 2
    assert result['class_names'] == (['DE','NDE'] if task == 'de_nde' else ['RR','RH'])
    for cohort in ['all','paired','raw_only']:
        assert result['test_cohorts'][cohort]['metrics']['n_samples'] > 0
    rows = list(csv.DictReader((run/'test_predictions.csv').open()))
    if task == 'rr_rh':
        assert 'probability_RR' in rows[0] and 'known_RH' in rows[0]
        assert any(abs(float(row['probability_RR']) + float(row['probability_RH']) - 1) > 1e-4 for row in rows)
    compare_experiments([run], tmp_path/'comparison.csv', cohort='all')
    if 'ssn' in model:
        assert result['config']['use_mixup'] is False


def test_suite_without_explicit_splits_and_seed(catalog, tmp_path):
    config = ExperimentConfig('suite', task='rr_rh', size=16, width=2, epochs=2, batch_size=4,
        device='cpu', augment=False, use_mixup=False, processed_dir=None, verbose=False)
    runs = run_experiments(catalog, [config], tmp_path/'runs')
    assert json.loads((runs[0]/'status.json').read_text())['status'] == 'completed'


def test_catalogue_parser():
    from clusterprep.categorise_PSZ2 import parse_classification
    assert parse_classification('DE') == [(50, 'DE')]
    names = [name for _, name in parse_classification('RR + cRR + RH + cRH')]
    assert set(names) == {'RR','cRR','RH','cRH','DE','cDE'}
    assert len(names) == len(set(names))
    with pytest.raises(ValueError, match='contradictoires'):
        parse_classification('RR + NDE')


def test_ignore_integration_and_masked_mixup(catalog, tmp_path):
    config = ExperimentConfig('ignore', task='rr_rh', candidate_policy='ignore', size=16,
        width=2, epochs=2, batch_size=4, device='cpu', augment=True,
        use_mixup=True, mixup_probability=1, processed_dir=None, verbose=False)
    runs = run_experiments(catalog, [config], tmp_path/'runs')
    result = json.loads((runs[0]/'metrics.json').read_text())
    assert result['test']['per_label']['RR']['n_known'] <= result['test']['n_samples']
    assert all('known_RR' in row for row in csv.DictReader((runs[0]/'test_predictions.csv').open()))


def test_reversed_class_order(catalog, tmp_path):
    config = ExperimentConfig('reversed', size=16, width=2, epochs=2, batch_size=4,
        device='cpu', augment=False, processed_dir=None, verbose=False)
    run = run_experiment(catalog, config, tmp_path/'runs', class_names=['NDE','DE'])
    results = json.loads((run/'metrics.json').read_text())
    assert results['class_names'] == ['NDE','DE']
    for row in csv.DictReader((run/'test_predictions.csv').open()):
        assert int(row['label']) == (0 if 'NDE' in catalog[row['key']]['labels'] else 1)


def test_wizard_multilabel_discovery(tmp_path):
    from clusterprep.config_wizard import create_config
    result = create_config(tmp_path/'wizard.json', print_fn=lambda *a: None,
                           input_fn=lambda prompt: 'rr_rh' if prompt.startswith('Tâche') else '')
    spec = json.loads(result.read_text())
    assert spec['class_names'] is None
    assert spec['experiments'][0]['task'] == 'rr_rh'


def test_processed_invalidation_preserved_mtime(catalog, tmp_path):
    import os
    from clusterprep.processed import load_processed
    info = catalog['c0.fits']
    load_processed(info, 'raw', 16, tmp_path/'cache')
    before = Path(info['raw_path']).stat()
    with fits.open(info['raw_path'], mode='update') as h:
        h[0].data[:] = 0
    os.utime(info['raw_path'], ns=(before.st_atime_ns, before.st_mtime_ns))
    result, status = load_processed(info, 'raw', 16, tmp_path/'cache', return_status=True)
    assert status == 'rebuilt' and np.nanmax(result['raw']) == 0


@pytest.mark.parametrize('unsafe', [False, True])
def test_downloader_archive_staging(tmp_path, unsafe):
    import io
    import tarfile
    from types import SimpleNamespace
    pytest.importorskip('requests')
    pytest.importorskip('bs4')
    from clusterprep import rsync_PSZ2 as downloader
    slug = 'test'
    dest = tmp_path/'fits'/slug
    dest.mkdir(parents=True)
    (dest/'keep.fits').write_bytes(b'old data')
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w') as stream:
        names = ['../escape.fits'] if unsafe else [f'{slug}{suffix}.fits' for suffix in ['', 'T25kpc', 'T50kpc', 'T100kpc', 'T25kpcSUB', 'T50kpcSUB', 'T100kpcSUB']]
        for name in names:
            member = tarfile.TarInfo(name)
            member.size = 4
            stream.addfile(member, io.BytesIO(b'fits'))
    page = SimpleNamespace(status_code=200, text='<a href="data.tar">data</a>')
    payload = SimpleNamespace(content=archive.getvalue(), raise_for_status=lambda: None)
    with patch.object(downloader, 'OUT_ROOT', tmp_path), patch.object(downloader, 'FITS_ROOT', tmp_path/'fits'), patch.object(downloader.requests, 'get', side_effect=[page, payload]):
        status, count = downloader.download_and_extract('https://example.invalid/test', slug, force=True)
    if unsafe:
        assert status == 'failed' and (dest/'keep.fits').read_bytes() == b'old data'
        assert not (tmp_path/'fits'/'escape.fits').exists()
    else:
        assert status == 'downloaded' and count == 7


def test_partial_only_validation_and_duplicate_rejection(catalog, tmp_path):
    from clusterprep.training import _validate_splits
    aliases = dict(catalog, alias=catalog['c0.fits'])
    with pytest.raises(ValueError, match='dupliqué'):
        _validate_splits(aliases, (['c0.fits','alias'], ['c1.fits'], ['c2.fits']), ['DE','NDE'])
    config = ExperimentConfig('partial', task='rr_rh', candidate_policy='ignore', size=16,
        width=2, epochs=2, batch_size=2, device='cpu', augment=False,
        use_mixup=False, processed_dir=None, verbose=False)
    splits = (['c0.fits','c1.fits','c2.fits','c6.fits'], ['c3.fits','c4.fits'], ['c9.fits','c10.fits'])
    run = run_experiment(catalog, config, tmp_path/'runs', splits=splits)
    result = json.loads((run/'metrics.json').read_text())
    assert result['test']['subset_accuracy'] is None
    assert 'val_RR_f1' in next(csv.DictReader((run/'history.csv').open()))
