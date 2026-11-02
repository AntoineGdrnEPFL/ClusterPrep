"""Subtype scalar targets and shared binary training, on synthetic inputs only."""
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from clusterprep import ClusterDataset, ExperimentConfig, run_experiment, build_model
from clusterprep.labels import task_target
from clusterprep.splits import eligible_for_task, split_data_keys
from clusterprep.training_core import build_criterion
from clusterprep.cross_validation import build_internal_folds
from test_transfer import single_thread


@pytest.mark.parametrize('labels,expected', [
    ({'RR'}, (1, 0)), ({'RH'}, (0, 1)), ({'NDE'}, (0, 0)),
    ({'RR', 'RH'}, (1, 1)), ({'DE'}, (None, None)),
    ({'cDE'}, (None, None)), ({'U'}, (None, None)),
    ({'cRR'}, (1, 0)), ({'cRH'}, (0, 1)),
    ({'RR', 'cRR'}, (1, 0)), ({'RR', 'cRH'}, (1, 1)),
])
def test_positive_targets(labels, expected):
    for task, target in zip(('rr_binary', 'rh_binary'), expected):
        info = {'labels': labels}
        assert eligible_for_task(info, task) == (target is not None)
        if target is None:
            with pytest.raises(ValueError):
                task_target(info, task)
        else:
            value, mask = task_target(info, task)
            assert value == target and isinstance(value, int) and mask is None


@pytest.mark.parametrize('labels,expected', [
    ({'cRR'}, (None, None)), ({'cRH'}, (None, None)),
    ({'RR', 'cRH'}, (1, None)), ({'RH', 'cRR'}, (None, 1)),
    ({'RR', 'cRR'}, (1, 0)), ({'RR', 'RH', 'cRR', 'cRH'}, (1, 1)),
])
def test_ignore_targets(labels, expected):
    for task, target in zip(('rr_binary', 'rh_binary'), expected):
        info = {'labels': labels}
        assert eligible_for_task(info, task, 'ignore') == (target is not None)
        if target is None:
            with pytest.raises(ValueError):
                task_target(info, task, 'ignore')
        else:
            assert task_target(info, task, 'ignore') == (target, None)


@pytest.mark.parametrize('task', ['rr_binary', 'rh_binary', 'de_nde', 'rr_rh'])
def test_contradictions(task):
    for info in ({'labels': {'NDE', 'RR'}}, {'class': 'RH', 'labels': {'RR'}}):
        with pytest.raises(ValueError):
            eligible_for_task(info, task)


def catalogue(n=60):
    return {str(i): {'cluster_id': str(i), 'labels': [{"RR"}, {"RH"}, {"NDE"}][i % 3],
                     'raw_path': 'unused', 'chandra_path': None} for i in range(n)}


def test_shared_splits_and_cv():
    info = catalogue()
    rr = split_data_keys(info, task='rr_binary')
    assert rr == split_data_keys(info, task='rh_binary')
    assert not set(rr[0]) & set(rr[1]) and not set(rr[1]) & set(rr[2])
    for task in ('rr_binary', 'rh_binary'):
        for keys in rr:
            assert {task_target(info[k], task)[0] for k in keys} == {0, 1}
    configs = [ExperimentConfig(name=t, task=t, cross_validation={'enabled': True, 'n_splits': 3})
               for t in ('rr_binary', 'rh_binary')]
    folds = [build_internal_folds(info, rr[0], c) for c in configs]
    assert [f['validation'] for f in folds[0]['folds']] == [f['validation'] for f in folds[1]['folds']]
    validations = [k for f in folds[0]['folds'] for k in f['validation']]
    assert len(validations) == len(set(validations)) == len(rr[0])


@pytest.mark.parametrize('task', ['rr_binary', 'rh_binary'])
@pytest.mark.parametrize('mode', ['raw', 'pair', 'multimodal'])
@pytest.mark.parametrize('transfer', [False, True])
def test_training_roundtrip(tmp_path, monkeypatch, task, mode, transfer):
    info = catalogue(18)
    for key, item in info.items():
        path = tmp_path / (key + '.fits')
        path.write_bytes(b'synthetic source identity')
        item.update(raw_path=str(path), chandra_path=str(path))
    def images(*args, **kwargs):
        return {'raw': np.arange(64, dtype=np.float32).reshape(8, 8),
                'chandra': np.ones((8, 8), dtype=np.float32),
                'chandra_mask': np.ones((8, 8), dtype=bool)}
    monkeypatch.setattr('clusterprep.dataset.preprocess', images)
    cfg = ExperimentConfig(name=task, task=task, mode=mode, epochs=1, width=2,
                           size=8, batch_size=3, device='cpu', augment=False,
                           processed_dir=None, verbose=False, use_scheduler=False)
    if transfer:
        from test_modular_transfer import save_source, plan
        source_path, source = save_source(tmp_path, cfg=replace(cfg, task="de_nde"))
        encoder = next(name for name in source.transfer_modules() if name != 'head')
        cfg = replace(cfg, transfer=plan(source_path, mappings=[
            {'source': 'source', 'from': encoder, 'to': encoder}]))
    cfg = ExperimentConfig(**json.loads(json.dumps(asdict(cfg))))
    assert isinstance(build_criterion(cfg, [6, 6]), nn.CrossEntropyLoss)
    ds = ClusterDataset(info, ['0', '1', '2'], ['DE', 'NDE'], mode=mode,
                        task=task, processed_dir=None, size=8, cache=True)
    assert ds[0]['label'].ndim == 0 and ds[0]['label'].dtype == torch.long
    splits = tuple([str(i) for i in range(a, a + 6)] for a in (0, 6, 12))
    run = run_experiment(info, cfg, tmp_path / 'runs', splits=splits)
    checkpoint = torch.load(run / 'best.pt', weights_only=False)
    model = build_model(cfg, 2)
    model.load_state_dict(checkpoint['model_state_dict'])
    metrics = json.loads((run / 'metrics.json').read_text())
    assert metrics['test']['positive_class'] == ('RR' if task == 'rr_binary' else 'RH')
    assert metrics['test']['roc_auc_macro'] is not None
    if transfer:
        assert metrics['independent_evaluation'] is False
        assert (run / 'transfer_splits_report.json').is_file()
    rows = list(csv.DictReader((run / 'test_predictions.csv').open()))
    assert len(rows) == 6 and all(r['annotations'] in {'RR', 'RH', 'NDE'} for r in rows)
    assert all(r['label'] in {'0', '1'} for r in rows)
    assert f"probability_{metrics['test']['positive_class']}" in rows[0]


def test_legacy_targets():
    assert task_target({'class': 'DE'}, 'de_nde') == (0, None)
    assert task_target({'class': 'NDE'}, 'de_nde') == (1, None)
    y, mask = task_target({'labels': {'RR', 'cRH'}}, 'rr_rh', 'ignore')
    assert y.tolist() == [1, 0] and mask.tolist() == [1, 0]


@pytest.mark.parametrize('task', ['rr_binary', 'rh_binary'])
def test_example_config(task):
    spec = json.loads((Path(__file__).parents[1] / 'configs' / f'{task}.json').read_text())
    assert ExperimentConfig(**spec['experiments'][0]).task == task


@pytest.mark.parametrize('task', ['rr_binary', 'rh_binary'])
@pytest.mark.parametrize('model,mode', [('image_cnn', 'raw'), ('dual_ssn', 'raw'),
                                       ('dual_encoder_cnn', 'multimodal')])
def test_encoder_transfer_preserves_fresh_head(tmp_path, task, model, mode):
    import copy
    from clusterprep import initialize_transfer
    from clusterprep.models import output_names_for_task
    from test_modular_transfer import save_source, plan
    cfg = ExperimentConfig(name='source', model=model, mode=mode, size=16, width=2)
    source_path, source = save_source(tmp_path, cfg=cfg)
    target = build_model(replace(cfg, task=task), 2)
    groups = target.transfer_modules()
    head = copy.deepcopy(groups['head'].state_dict())
    source_groups = source.transfer_modules()
    encoder = next(name for name in source_groups if name != 'head')
    initialize_transfer(target, plan(source_path, mappings=[
        {'source': 'source', 'from': encoder, 'to': encoder}]), build_model)
    assert all(torch.equal(value, groups['head'].state_dict()[name]) for name, value in head.items())
    assert all(torch.equal(value, groups[encoder].state_dict()[name])
               for name, value in source_groups[encoder].state_dict().items())
    assert output_names_for_task(task)[1] == ('RR' if task == 'rr_binary' else 'RH')


def test_binary_gallery_and_positive_scores(tmp_path, monkeypatch):
    import importlib.util
    import pandas as pd
    from types import SimpleNamespace
    import matplotlib.pyplot as plt
    spec = importlib.util.spec_from_file_location('binary_analysis', Path(__file__).parents[1] / 'notebooks' / 'training_analysis.py')
    module = importlib.util.module_from_spec(spec)
    import sys
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    frame = pd.DataFrame({'key': ['a', 'b', 'c', 'd'], 'cluster_id': ['a', 'b', 'c', 'd'],
                          'label': [1, 0, 0, 1], 'prediction': [1, 0, 1, 0],
                          'probability_non_RR': [.1, .9, .1, .9],
                          'probability_RR': [.9, .1, .9, .1], 'has_chandra': [True] * 4})
    scores = module.gain_scores(frame, ['non_RR', 'RR'])
    assert scores['f1_positive'] == .5 and 'f1_de' not in scores
    monkeypatch.setattr('clusterprep.multilabel_analysis._prepare_cluster_modalities',
                        lambda *a, **kw: (np.arange(64).reshape(8, 8), np.arange(64).reshape(8, 8)))
    run = SimpleNamespace(classes=['non_RR', 'RR'], config={'task': 'rr_binary'},
                          predictions=lambda *args: frame)
    for chandra_only in (False, True):
        figs = module.plot_binary_galleries(run, processed_root=tmp_path, chandra_only=chandra_only)
        assert set(figs) == {'TP', 'TN', 'FP', 'FN'}
        for fig in figs.values():
            assert 'rr_binary' in fig._suptitle.get_text()
            plt.close(fig)


def test_incompatible_split_rejected(tmp_path):
    info = catalogue(9)
    info['0']['labels'] = {'DE'}
    cfg = ExperimentConfig(name='bad_split', task='rr_binary')
    with pytest.raises(ValueError, match='non éligible'):
        run_experiment(info, cfg, tmp_path, splits=(['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']))


def test_transfer_rejects_source_training_exposure(tmp_path):
    from test_modular_transfer import plan
    info = catalogue(9)
    (tmp_path / 'splits.json').write_text(json.dumps({
        'train': ['source'], 'validation': [], 'test': [],
        'samples': {'source': {'cluster_id': '6'}}}))
    cfg = ExperimentConfig(name='leak', task='rh_binary', transfer=plan(tmp_path / 'best.pt'))
    with pytest.raises(ValueError, match='Fuite'):
        run_experiment(info, cfg, tmp_path / 'runs',
                       splits=(['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']))


def test_cli_discovers_subtypes_from_legacy_json(tmp_path, monkeypatch):
    from clusterprep.__main__ import main
    seen = {}
    def discover(path, classes):
        seen['classes'] = classes
        return catalogue()
    monkeypatch.setattr('clusterprep.__main__.get_data_info_dict', discover)
    monkeypatch.setattr('clusterprep.training.run_experiments', lambda *a, **kw: [tmp_path / 'run'])
    monkeypatch.setattr('clusterprep.training.compare_experiments', lambda *a, **kw: [])
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'data_dir': 'unused', 'output_dir': str(tmp_path / 'out'),
                                'class_names': ['DE', 'NDE'],
                                'experiments': [{'name': 'rr', 'task': 'rr_binary'}]}))
    main([str(path), '--quiet'])
    assert seen['classes'] is None
