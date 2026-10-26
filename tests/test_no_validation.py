"""Fixed-budget training must never select a checkpoint using test data."""
import csv
import json
from dataclasses import asdict, replace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.wcs import WCS
from torch.utils.data import DataLoader

from clusterprep import ExperimentConfig, Trainer, TransferPhase, run_experiments, plot_history
from clusterprep.config_wizard import create_config
from clusterprep.splits import split_data_keys
from test_transfer import TinyBN, single_thread


def inventory():
    return {str(i): dict(cluster_id=str(i), labels={'NDE'} if i % 4 == 0 else
            {'DE', 'RR'} if i % 4 == 1 else {'DE', 'RH'} if i % 4 == 2 else {'DE', 'RR', 'RH'})
            for i in range(40)}


@pytest.mark.parametrize('task', ['de_nde', 'rr_rh'])
@pytest.mark.parametrize('folds', [None, 5])
def test_split_keeps_test_and_moves_whole_pool_to_train(task, folds):
    info = inventory()
    original = split_data_keys(info, task=task, n_folds=folds)
    for options in ({'use_val': False}, {'val_size': 0}):
        train, val, test = split_data_keys(info, task=task, n_folds=folds, **options)
        assert val == []
        assert test == original[2]
        assert set(train) == set(original[0] + original[1])
        assert not set(train) & set(test)


@pytest.mark.parametrize('options', [{'use_val': 'false'}, {'val_size': -1}, {'val_size': float('nan')}])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        split_data_keys(inventory(), **options)


def configuration(**kwargs):
    return ExperimentConfig('fixed', size=8, width=2, batch_size=4, epochs=3,
        device='cpu', augment=False, processed_dir=None, verbose=False,
        patience=1, min_delta=100, use_scheduler=True, **kwargs)


@pytest.mark.parametrize('phased', [False, True])
def test_fit_never_evaluates_and_retains_last_weights(tmp_path, phased):
    cfg = configuration(training_phases=[
        TransferPhase('head', 2, {'head': .01}),
        TransferPhase('all', 3, {'radio': .001, 'head': .01}),
    ] if phased else [])
    model = TinyBN()
    loader = DataLoader([{'raw': torch.randn(1, 8, 8), 'label': torch.tensor(i % 2)}
                         for i in range(8)], batch_size=4)
    trainer = Trainer(model, cfg, ['DE', 'NDE'], [4, 4], steps_per_epoch=len(loader))
    last = {}
    original = trainer.train_one_epoch
    def train(*args, **kwargs):
        result = original(*args, **kwargs)
        last.update({k: v.clone() for k, v in model.state_dict().items()})
        return result
    with patch.object(trainer, 'evaluate', side_effect=AssertionError('No validation evaluation')), \
         patch.object(trainer, 'train_one_epoch', side_effect=train):
        history = (trainer.fit_transfer(loader, None, run_dir=tmp_path) if phased
                   else trainer.fit(loader, run_dir=tmp_path))
    count = 5 if phased else 3
    assert len(history) == count
    assert trainer.best_epoch == count
    assert not trainer.stopped_early
    assert trainer.stopping.best is None
    assert all(r['validation'] is None for r in history)
    checkpoint = torch.load(tmp_path / 'best.pt', weights_only=True)
    assert checkpoint['epoch'] == count
    assert checkpoint['selection'] == 'last_epoch'
    assert checkpoint['monitor'] is None and checkpoint['score'] is None
    for key in last:
        torch.testing.assert_close(model.state_dict()[key], last[key], rtol=0, atol=0)
        torch.testing.assert_close(checkpoint['model_state_dict'][key], last[key], rtol=0, atol=0)
    figure = plot_history(tmp_path)
    assert all(len(ax.lines) >= 1 for ax in figure.axes)
    plt.close(figure)


@pytest.mark.parametrize('task', ['de_nde', 'rr_rh'])
@pytest.mark.parametrize('interface', ['explicit', 'cli', 'empty'])
def test_suite_artifacts_and_only_final_test_evaluation(tmp_path, task, interface):
    info = inventory()
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crpix = [4, 4]
    wcs.wcs.cdelt = [-.01, .01]
    path = tmp_path / 'image.fits'
    fits.PrimaryHDU(np.arange(64, dtype='float32').reshape(8, 8), wcs.to_header()).writeto(path)
    for item in info.values():
        item.update(raw_path=str(path), chandra_path=None)
    splits = split_data_keys(info, task=task)
    cfg = configuration(task=task)
    calls = []
    original = Trainer.evaluate
    def evaluate(self, loader, description=None):
        calls.append(description)
        return original(self, loader, description)
    with patch.object(Trainer, 'evaluate', evaluate):
        if interface == 'cli':
            from clusterprep.__main__ import main
            config_path = tmp_path / 'config.json'
            config_path.write_text(json.dumps({
                'data_dir': str(tmp_path), 'output_dir': str(tmp_path / 'runs'),
                'split': {'val_size': 0}, 'experiments': [asdict(cfg)],
            }))
            with patch('clusterprep.__main__.get_data_info_dict', return_value=info):
                main([str(config_path)])
            run = next(p for p in (tmp_path / 'runs').iterdir() if p.is_dir())
            assert list((tmp_path / 'runs').glob('comparison-*.csv'))
        elif interface == 'empty':
            from clusterprep import run_experiment
            run = run_experiment(info, cfg, tmp_path / 'runs',
                                 splits=(splits[0] + splits[1], [], splits[2]))
        else:
            run = run_experiments(info, [cfg], tmp_path / 'runs', splits=splits, use_val=False)[0]
    assert calls == ['Test final']
    result = json.loads((run / 'metrics.json').read_text())
    assert result['split_counts'] == {'train': 32, 'validation': 0, 'test': 8}
    assert result['epochs_ran'] == 3 and result['best_epoch'] == 3
    assert result['validation'] is None and result['monitor_best'] is None
    assert result['selection'] == 'last_epoch' and not result['stopped_early']
    assert result['config']['use_early_stopping'] is False
    manifest = json.loads((run / 'splits.json').read_text())
    assert set(manifest['train']) == set(splits[0] + splits[1])
    if interface != 'cli':
        assert manifest['train'] == splits[0] + splits[1]
    assert manifest['test'] == splits[2]
    assert not list(csv.DictReader((run / 'validation_predictions.csv').open()))
    assert len(list(csv.DictReader((run / 'test_predictions.csv').open()))) == 8
    if task == 'rr_rh':
        from clusterprep.multilabel_analysis import load_run, plot_learning_history
        loaded = load_run(run)
        assert loaded.predictions['validation'].targets.shape == (0, 2)
        figure = plot_learning_history(loaded)
        plt.close(figure)
        assert result['threshold_selection']['method'] == 'fixed'
        assert result['config']['thresholds'] == cfg.thresholds
        with pytest.raises(ValueError, match='threshold_selection'):
            run_experiments(info, [replace(cfg, threshold_selection='f1')], tmp_path, splits=splits, use_val=False)
    with pytest.raises(ValueError, match='Fuite'):
        run_experiments(info, [cfg], tmp_path, splits=(splits[0], splits[0], splits[2]), use_val=False)


def test_wizard_zero_validation(tmp_path):
    path = tmp_path / 'config.json'
    create_config(path, input_fn=lambda prompt: '0' if prompt.startswith('Fraction validation') else '',
                  print_fn=lambda *_: None)
    spec = json.loads(path.read_text())
    assert spec['split']['val_size'] == 0
    assert spec['experiments'][0]['use_early_stopping'] is False
    assert spec['experiments'][0]['threshold_selection'] == 'fixed'
