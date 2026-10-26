"""CV invariants on synthetic images; never need the user's scientific dataset."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.wcs import WCS

from clusterprep import (CrossValidationConfig, ExperimentConfig, Trainer, TransferPhase,
                         build_internal_folds, aggregate_curves, run_experiment, run_experiments,
                         build_model, compare_experiments, plot_history)
from clusterprep.cross_validation import audit_initialization
from clusterprep.training import compute_task_counts
from test_transfer import TinyBN, single_thread, source_run


def cfg(**kwargs):
    defaults = dict(name='cv', epochs=3, size=8, width=2, batch_size=4, device='cpu',
                    processed_dir=None, augment=False, verbose=False, use_scheduler=False,
                    use_mixup=False, patience=1, min_delta=100,
                    cross_validation={'enabled': True, 'n_splits': 3, 'max_epochs': 3})
    defaults.update(kwargs)
    return ExperimentConfig(**defaults)


def inventory(n=40):
    return {str(i): {'cluster_id': f'cluster_{i:03}', 'labels': (
        {'NDE'} if i % 4 == 0 else {'DE', 'RR'} if i % 4 == 1 else
        {'DE', 'RH'} if i % 4 == 2 else {'DE', 'RR', 'RH'})} for i in range(n)}


@pytest.fixture
def images(tmp_path):
    info = inventory()
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crpix = [4, 4]
    wcs.wcs.cdelt = [-.01, .01]
    path = tmp_path / 'image.fits'
    fits.PrimaryHDU(np.random.default_rng(42).normal(size=(8, 8)).astype('float32'), wcs.to_header()).writeto(path)
    for i, item in enumerate(info.values()):
        item.update(raw_path=str(path), chandra_path=str(path) if i % 8 < 4 else None)
    return info


def splits():
    return ([str(i) for i in range(24)], [str(i) for i in range(24, 32)],
            [str(i) for i in range(32, 40)])


@pytest.mark.parametrize('task,splitter', [('de_nde', 'StratifiedKFold'), ('rr_rh', 'MultilabelStratifiedKFold')])
@pytest.mark.parametrize('shuffle', [True, False])
def test_folds_reproducible_exhaustive_cluster_stratification(task, splitter, shuffle):
    info = inventory()
    c = cfg(task=task, cross_validation={'enabled': True, 'n_splits': 3, 'shuffle': shuffle})
    first = build_internal_folds(info, list(info), c)
    assert first == build_internal_folds(info, list(reversed(info)), c)
    assert first['strategy'] == splitter
    validation = []
    for fold in first['folds']:
        assert not set(fold['train_cluster_ids']) & set(fold['validation_cluster_ids'])
        assert set(fold['train'] + fold['validation']) == set(info)
        validation.extend(fold['validation'])
        assert fold['distribution']['validation']['n_clusters'] == len(fold['validation'])
    assert len(validation) == len(info) and set(validation) == set(info)


def test_unknown_rr_rh_masks_never_become_stratification_negatives():
    info = inventory()
    info['partial'] = {'cluster_id': 'partial', 'labels': {'cRR', 'RH'}}
    c = cfg(task='rr_rh', candidate_policy='ignore')
    with pytest.warns(UserWarning, match='Partially known'):
        plan = build_internal_folds(info, list(info), c)
    assert plan['partially_known_clusters'] == ['partial']
    assert plan['distribution']['unknown'] == {'RR': 1, 'RH': 0}
    assert plan['distribution']['RR-RH-'] == 10
    assert sum('partial' in f['validation'] for f in plan['folds']) == 1


def test_fold_preflight_errors():
    info = inventory(4)
    with pytest.raises(ValueError, match='reduce n_splits'):
        build_internal_folds(info, list(info), cfg())
    info = inventory()
    info['alias'] = copy.deepcopy(info['0'])
    with pytest.raises(ValueError, match='unique cluster'):
        build_internal_folds(info, list(info), cfg())
    with pytest.raises(ValueError, match='incompatible'):
        build_internal_folds(inventory(), list(inventory()), cfg(cross_validation={'enabled': True, 'strategy': 'multilabel'}))


def test_weighted_epoch_selection_direction_dispersion_and_failures():
    curves = [[{'epoch': i+1, 'validation': {'loss': loss, 'macro_f1': loss}} for i, loss in enumerate(values)]
              for values in ([1., 3., 3.], [5., 1., 1.])]
    result = aggregate_curves(curves, [1, 3])
    assert result['selected_epoch'] == 2
    assert result['curves'][0]['mean'] == 4
    assert result['curves'][0]['std'] == pytest.approx(np.sqrt(3))
    assert aggregate_curves(curves, [1, 3], 'f1_macro', 'max')['selected_epoch'] == 1
    with pytest.raises(ValueError, match='Incomplete'):
        aggregate_curves([curves[0][:-1], curves[1]], [1, 3])
    curves[1][0]['validation']['loss'] = None
    with pytest.raises(ValueError, match='Undefined'):
        aggregate_curves(curves, [1, 3])


@pytest.mark.parametrize('task', ['de_nde', 'rr_rh'])
def test_end_to_end_clean_states_no_test_access_and_final_reload(images, tmp_path, task):
    states, model_ids, eval_calls = [], [], []
    models = []
    def factory(config, num_classes):
        model = build_model(config, num_classes)
        states.append(copy.deepcopy(model.state_dict()))
        models.append(model)
        model_ids.append(id(model))
        return model
    c = cfg(task=task, augment=True, threshold_selection='f1' if task == 'rr_rh' else 'fixed')
    original = Trainer.evaluate
    def evaluate(self, loader, description=None):
        assert not loader.dataset.augment
        keys = set(loader.dataset.keys)
        if keys & set(splits()[2]):
            assert keys == set(splits()[2])
            checkpoint = next((tmp_path / 'runs').glob('*/final/best.pt'))
            saved = torch.load(checkpoint, weights_only=True)
            for k, v in self.model.state_dict().items():
                torch.testing.assert_close(v.cpu(), saved['model_state_dict'][k], rtol=0, atol=0)
            eval_calls.append('test')
        else:
            eval_calls.append('validation')
        return original(self, loader, description)
    with patch.object(Trainer, 'evaluate', evaluate):
        final = run_experiment(images, c, tmp_path / 'runs', splits=splits(), model_factory=factory)
    assert len(states) == 4 and len(set(model_ids)) == 4
    for state in states[1:]:
        for k in state:
            torch.testing.assert_close(state[k], states[0][k], rtol=0, atol=0)
    assert eval_calls.count('test') == 1
    assert eval_calls.count('validation') == 9
    root = final.parent
    selection = json.loads((root / 'cv/epoch_selection.json').read_text())
    manifest = json.loads((final / 'splits.json').read_text())
    assert manifest['train'] == splits()[0] + splits()[1]
    assert manifest['validation'] == [] and manifest['test'] == splits()[2]
    result = json.loads((final / 'metrics.json').read_text())
    assert result['epochs_ran'] == selection['selected_epoch']
    assert result['validation'] is None and result['selection'] == 'last_epoch'
    assert result['test']['n_samples'] == 8
    for fold_dir in (root / 'cv/trial_0').iterdir():
        metadata = json.loads((fold_dir / 'config.json').read_text())
        fold = json.loads((fold_dir / 'splits.json').read_text())
        expected = compute_task_counts(images, fold['train'], task, 'positive')
        assert list(metadata['train_class_counts'].values()) == expected
        assert fold['test'] == []
        assert not set(fold['samples']) & set(splits()[2])
        assert json.loads((fold_dir / 'metrics.json').read_text())['epochs_ran'] == 3
        assert not (fold_dir / 'test_predictions.csv').exists()
    checkpoint = torch.load(final / 'best.pt', weights_only=True)
    assert checkpoint['cross_validation']['selection'] == selection
    assert len(json.loads((root / 'cv/oof_predictions.json').read_text())) == 32
    assert (root / 'cv/cv_curves.png').stat().st_size > 0
    assert compare_experiments([final], cohort='all')[0]['n_samples'] == 8
    import matplotlib.pyplot as plt
    plt.close(plot_history(final))
    if task == 'rr_rh':
        from clusterprep.multilabel_analysis import load_run
        loaded = load_run(final)
        assert loaded.thresholds == result['config']['thresholds']
        assert checkpoint['evaluation']['provenance'].startswith('pooled internal')


def test_coupled_phase_grid_exact_trajectories(images, tmp_path):
    phases = [TransferPhase('head', 2, {'head': .01}), TransferPhase('all', 2, {'radio': .001, 'head': .01})]
    candidates = [{'head': 1, 'all': 2}, {'head': 2, 'all': 1}]
    c = cfg(training_phases=phases, cross_validation={'enabled': True, 'n_splits': 2, 'phase_durations': candidates})
    states = []
    def factory(*_):
        model = TinyBN()
        states.append(copy.deepcopy(model.state_dict()))
        return model
    final = run_experiment(images, c, tmp_path / 'runs', splits=splits(), model_factory=factory)
    assert len(states) == 5
    for state in states[1:]:
        for k in state:
            torch.testing.assert_close(state[k], states[0][k], rtol=0, atol=0)
    summary = json.loads((final.parent / 'cv/cv_summary.json').read_text())
    chosen = summary['selection']['phase_durations']
    assert chosen in candidates
    for trial in summary['trials']:
        assert trial['score'] == trial['aggregation']['curves'][-1]['mean']
        for relative in trial['fold_paths']:
            checkpoint = torch.load(final.parent / relative / 'best.pt', weights_only=True)
            assert checkpoint['epoch'] == 3 and checkpoint['phase'] == 'all'
            assert checkpoint['selection'] == 'last_epoch'
    report = json.loads((final / 'training_phases.json').read_text())
    assert {p['name']: p['epochs_ran'] for p in report['phases']} == chosen
    assert report['selected_phase'] == 'all'


@pytest.mark.parametrize('refit,evaluate', [(False, True), (True, False)])
def test_cv_only_and_final_without_test(images, tmp_path, refit, evaluate):
    c = cfg(cross_validation={'enabled': True, 'n_splits': 2, 'max_epochs': 1,
                            'retrain_full_train': refit, 'evaluate_test_after_retrain': evaluate})
    original = Trainer.evaluate
    def validation_only(self, loader, description=None):
        assert not set(loader.dataset.keys) & set(splits()[2])
        return original(self, loader, description)
    with patch.object(Trainer, 'evaluate', validation_only):
        path = run_experiment(images, c, tmp_path / 'runs', splits=splits())
    root = path.parent if refit else path
    status = json.loads((root / 'status.json').read_text())
    assert status['status'] == 'completed' and status['test_evaluated'] is False
    assert (root / 'final/best.pt').exists() == refit


def test_cli_one_command_and_disabled_regression(images, tmp_path):
    from clusterprep.__main__ import main
    path = tmp_path / 'config.json'
    c = cfg(cross_validation={'enabled': True, 'n_splits': 2, 'max_epochs': 1})
    path.write_text(json.dumps({'data_dir': str(tmp_path), 'output_dir': str(tmp_path / 'runs'),
                               'experiments': [asdict(c)], 'split': {'splits': splits()}}))
    with patch('clusterprep.__main__.get_data_info_dict', return_value=images):
        main([str(path)])
    assert len(list((tmp_path / 'runs').glob('*/final/test_predictions.csv'))) == 1
    assert len(list((tmp_path / 'runs').glob('comparison-*.csv'))) == 1
    disabled = replace(c, cross_validation={'enabled': False})
    a = run_experiment(images, disabled, tmp_path / 'old', splits=splits())
    b = run_experiment(images, replace(c, cross_validation=None), tmp_path / 'old', splits=splits())
    assert (a / 'test_predictions.csv').read_text() == (b / 'test_predictions.csv').read_text()
    assert json.loads((a / 'metrics.json').read_text())['best_epoch'] == 1


def test_failed_fold_never_evaluates_test_and_records_status(images, tmp_path):
    with patch.object(Trainer, 'train_one_epoch', side_effect=RuntimeError('deliberate failure')):
        with pytest.raises(RuntimeError, match='deliberate'):
            run_experiment(images, cfg(), tmp_path / 'runs', splits=splits())
    root = next((tmp_path / 'runs').iterdir())
    status = json.loads((root / 'status.json').read_text())
    assert status['status'] == 'failed' and status['fold'] == 0
    assert json.loads((root / 'cv/trial_0/fold_0/status.json').read_text())['status'] == 'failed'
    assert not (root / 'final').exists()
    assert not (root / 'cv/cv_summary.json').exists()


def save_source(tmp_path, exposed, transfer=None):
    source = tmp_path / 'source'
    source.mkdir(exist_ok=True)
    c = replace(cfg(), cross_validation=None, transfer=transfer)
    model = build_model(c, 2)
    torch.save({'config': asdict(c), 'class_names': ['DE', 'NDE'], 'model_state_dict': model.state_dict()}, source / 'best.pt')
    (source / 'splits.json').write_text(json.dumps({'train': exposed, 'validation': [], 'test': [],
        'samples': {k: {'cluster_id': k} for k in exposed}}))
    return source, model


def modular(source):
    return {'enabled': True, 'sources': {'s': {'checkpoint': str(source / 'best.pt')}},
            'mappings': [{'source': 's', 'from': 'radio', 'to': 'radio'}]}


def test_pretraining_audit_overlap_unknown_and_disjoint(images, tmp_path):
    info = images
    train, validation, test = splits()
    source, _ = save_source(tmp_path, ['cluster_000'])
    c = cfg(transfer=modular(source))
    folds = build_internal_folds(info, train + validation, c)['folds']
    report = audit_initialization(c, info, train + validation, test, folds)
    assert report['sources'][0]['external_train_overlap'] == ['cluster_000']
    assert not report['verified_no_overlap']
    with pytest.raises(ValueError, match='pretraining used'):
        run_experiment(info, c, tmp_path / 'runs', splits=splits())
    save_source(tmp_path, ['cluster_032'])
    with pytest.raises(ValueError, match='external test'):
        run_experiment(info, replace(c, cross_validation=replace(c.cross_validation, source_provenance_policy='warn')),
                       tmp_path / 'runs', splits=splits())
    save_source(tmp_path, ['external'])
    report = audit_initialization(c, info, train + validation, test, folds)
    assert report['verified_no_overlap']
    (source / 'splits.json').unlink()
    report = audit_initialization(c, info, train + validation, test, folds)
    assert report['sources'][0]['provenance'] == 'unknown'


def test_transfer_reloads_same_source_each_fold_and_final(images, tmp_path):
    source, original = save_source(tmp_path, ['external'])
    c = cfg(transfer=modular(source), cross_validation={'enabled': True, 'n_splits': 2, 'max_epochs': 1})
    initial = []
    from clusterprep.training import initialize_transfer
    def spy(model, config, factory):
        result = initialize_transfer(model, config, factory)
        initial.append(copy.deepcopy(model.encoder.state_dict()))
        return result
    with patch('clusterprep.training.initialize_transfer', spy):
        final = run_experiment(images, c, tmp_path / 'runs', splits=splits())
    assert len(initial) == 3
    for state in initial:
        for k in state:
            torch.testing.assert_close(state[k], original.encoder.state_dict()[k], rtol=0, atol=0)
    assert json.loads((final / 'transfer.json').read_text())['mappings'][0]['status'] == 'transferred'


@pytest.mark.parametrize('model,mode', [
    ('cnn', 'raw'), ('cnn', 'multimodal'), ('cnn', 'pair'),
    ('fusion', 'multimodal'), ('fusion', 'pair'),
    ('image_cnn', 'raw'), ('image_cnn', 'multimodal'), ('image_cnn', 'pair'),
    ('dual_encoder_cnn', 'raw'), ('dual_encoder_cnn', 'multimodal'), ('dual_encoder_cnn', 'pair'),
    ('dual_ssn', 'raw'), ('dual_ssn', 'multimodal'), ('dual_ssn', 'pair'),
    ('dual_encoder_ssn', 'raw'), ('dual_encoder_ssn', 'multimodal'), ('dual_encoder_ssn', 'pair'),
    ('spatial_difference_fusion', 'multimodal'), ('spatial_difference_fusion', 'pair'),
])
def test_architectures_modalities_use_same_cv_engine(images, tmp_path, model, mode):
    if 'ssn' in model:
        pytest.importorskip('kymatio')
    c = cfg(model=model, mode=mode, processed_dir=str(tmp_path / 'processed'),
            cross_validation={'enabled': True, 'n_splits': 2, 'max_epochs': 1})
    final = run_experiment(images, c, tmp_path / 'runs', splits=splits())
    result = json.loads((final / 'metrics.json').read_text())
    assert result['epochs_ran'] == 1
    assert result['split_counts'] == {'train': 16 if mode == 'pair' else 32,
                                      'validation': 0, 'test': 4 if mode == 'pair' else 8}
    assert (final / 'best.pt').exists()


def test_legacy_transfer_resolves_external_split_then_searches_phases(source_run, tmp_path):
    from test_transfer import config, transfer
    source, target = source_run
    c = config(task='rr_rh', transfer=transfer(source), cross_validation={
        'enabled': True, 'n_splits': 2, 'phase_durations': [{'head': 1, 'full': 1}],
        'source_provenance_policy': 'warn',
    })
    with pytest.warns(UserWarning, match='pretraining used'):
        final = run_experiment(target, c, tmp_path / 'cv_runs')
    manifest = json.loads((final / 'splits.json').read_text())
    ids = lambda part: {str(manifest['samples'][k]['cluster_id']) for k in manifest[part]}
    assert set(map(str, range(18, 24))) <= ids('test')
    assert not ids('train') & set(map(str, range(18, 24)))
    assert json.loads((final / 'metrics.json').read_text())['epochs_ran'] == 2
    audit = json.loads((final.parent / 'cv/initialization_audit.json').read_text())
    assert audit['verified_no_overlap'] is False


def test_optimizer_and_scheduler_start_fresh_for_each_fold_and_final(images, tmp_path):
    optimizers, schedulers = [], []
    original = Trainer.__init__
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        assert not self.optimizer.state
        assert self.scheduler.last_epoch == 0
        optimizers.append(self.optimizer)
        schedulers.append(self.scheduler)
    c = cfg(use_scheduler=True, cross_validation={'enabled': True, 'n_splits': 2, 'max_epochs': 2})
    with patch.object(Trainer, '__init__', initialize):
        final = run_experiment(images, c, tmp_path / 'runs', splits=splits())
    assert len({id(o) for o in optimizers}) == 3
    assert len({id(s) for s in schedulers}) == 3
    epochs = json.loads((final / 'metrics.json').read_text())['epochs_ran']
    assert schedulers[-1].last_epoch == epochs * 8


def test_changed_source_aborts_search_before_test(images, tmp_path):
    source, _ = save_source(tmp_path, ['external'])
    c = cfg(transfer=modular(source), cross_validation={'enabled': True, 'n_splits': 2, 'max_epochs': 1})
    original = Trainer.fit
    def fit(self, *args, **kwargs):
        history = original(self, *args, **kwargs)
        with (source / 'best.pt').open('ab') as stream:
            stream.write(b'changed')
        return history
    with patch.object(Trainer, 'fit', fit), pytest.raises(ValueError, match='source changed'):
        run_experiment(images, c, tmp_path / 'runs', splits=splits())
    root = next((tmp_path / 'runs').iterdir())
    assert json.loads((root / 'status.json').read_text())['status'] == 'failed'
    assert not (root / 'final').exists()


def test_ancestral_leakage_detected_with_bare_weights_and_sidecar_config(images, tmp_path):
    ancestor_dir = tmp_path / 'ancestor'
    ancestor_dir.mkdir()
    ancestor, _ = save_source(ancestor_dir, ['cluster_032'])
    source, model = save_source(tmp_path, ['external'], transfer=modular(ancestor))
    payload = torch.load(source / 'best.pt', weights_only=True)
    (source / 'config.json').write_text(json.dumps({'config': payload['config']}))
    torch.save(model.state_dict(), source / 'best.pt')
    c = cfg(transfer=modular(source))
    plan = build_internal_folds(images, splits()[0] + splits()[1], c)
    report = audit_initialization(c, images, splits()[0] + splits()[1], splits()[2], plan['folds'])
    assert len(report['sources']) == 2
    assert report['external_test_leakage']
    assert report['sources'][1]['external_test_overlap'] == ['cluster_032']


def test_bare_config_path_and_embedded_provenance_are_supported(images, tmp_path):
    source, _ = save_source(tmp_path, ['external'])
    payload = torch.load(source / 'best.pt', weights_only=True)
    payload['split_provenance'] = json.loads((source / 'splits.json').read_text())
    torch.save(payload, source / 'best.pt')
    (source / 'splits.json').unlink()
    c = cfg(transfer=modular(source))
    plan = build_internal_folds(images, splits()[0] + splits()[1], c)
    report = audit_initialization(c, images, splits()[0] + splits()[1], splits()[2], plan['folds'])
    assert report['verified_no_overlap'] and report['sources'][0]['provenance'] == 'embedded'
    (source / 'config.json').write_text(json.dumps(payload['config']))
    m = modular(source)
    m['sources']['s']['config_path'] = str(source / 'config.json')
    report = audit_initialization(replace(c, transfer=m), images, splits()[0] + splits()[1], splits()[2], plan['folds'])
    assert report['sources'][0]['provenance'] == 'embedded'
    assert report['verified_no_overlap']
