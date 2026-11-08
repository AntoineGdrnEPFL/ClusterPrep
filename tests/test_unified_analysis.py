"""Artifact contracts, statistical guards, and executable notebook regression tests."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from clusterprep import analysis as a
from clusterprep.analysis.comparison import score_run


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def make_run(tmp_path):
    def make(name='a', task='de_nde', fold=None, model='cnn', thresholds=True, transfer=None):
        path = tmp_path / name
        path.mkdir(parents=True)
        labels = ['RR', 'RH'] if task == 'rr_rh' else ['DE', 'NDE']
        config = dict(model=model, mode='raw', task=task, name='experiment', seed=42,
                      candidate_policy='positive', transfer=transfer)
        if thresholds:
            config['thresholds'] = {'RR': .7, 'RH': .3}
        write(path/'config.json', dict(config=config, class_names=labels))
        write(path/'metrics.json', dict(class_names=labels, test={'loss': .123}, epochs_ran=2))
        write(path/'status.json', {'status': 'completed'})
        samples = {k: dict(cluster_id=k.upper(), labels=[]) for k in ['t', 'v', 'a', 'b', 'c', 'd']}
        # Unknown annotations need not be reconstructed for binary tasks.
        if task == 'rr_rh':
            for k, lab in zip(['a','b','c','d'], [['RR'], ['RH'], ['RR','RH'], ['NDE']]):
                samples[k]['labels'] = lab
        write(path/'splits.json', dict(train=['t'], validation=['v'], test=['a','b','c','d'],
              samples=samples, split_config={'n_folds': 3 if fold is not None else None, 'fold_index': fold, 'split_seed': 42}))
        rows = dict(key=['a','b','c','d'], cluster_id=['A','B','C','D'], has_chandra=[False]*4)
        if task == 'rr_rh':
            rows.update(label_RR=[1,0,1,0], label_RH=[0,1,1,0], known_RR=[True]*4, known_RH=[True]*4,
                        probability_RR=[.8,.1,.6,.2], probability_RH=[.1,.8,.4,.2])
        else:
            rows.update(label=[0,0,1,1], prediction=[0,1,1,1],
                        probability_DE=[.9,.4,.2,.1], probability_NDE=[.1,.6,.8,.9])
        pd.DataFrame(rows).to_csv(path/'test_predictions.csv', index=False)
        (path/'history.jsonl').write_text('\n'.join(json.dumps(dict(epoch=i, train={'loss': 1/i}, validation={'loss': 2/i})) for i in [1,2]))
        return path
    return make


def test_recursive_empty_incomplete_invalid_duplicates(tmp_path, make_run):
    assert a.discover_runs(tmp_path)[0] == []
    path = make_run('nested/deep/run')
    (tmp_path/'alias').symlink_to(path, target_is_directory=True)
    runs, _ = a.discover_runs(tmp_path)
    assert len(runs) == 1
    assert not a.discover_runs(tmp_path, recursive=False)[0]
    assert len(a.discover_runs(path, recursive=False)[0]) == 1
    (path/'metrics.json').write_text('{broken')
    runs, issues = a.discover_runs(tmp_path)
    assert runs[0].status == 'incomplete'
    assert issues.issue.str.contains('metrics.json').any()
    (path/'status.json').unlink()
    assert a.load_run(path).status == 'unverified'
    bad = tmp_path/'bad'; bad.mkdir(); (bad/'config.json').write_text('[]')
    assert len(a.discover_runs(tmp_path)[1]) > len(issues)


def test_binary_scores_and_unknown_threshold(make_run):
    run = a.load_run(make_run())
    score, source = score_run(run)
    assert source == 'recomputed'
    assert score['accuracy'] == .75
    assert score['macro_f1'] == pytest.approx((2/3 + 4/5)/2)
    assert score['loss'] == .123
    multi = a.load_run(make_run('multi', task='rr_rh', thresholds=False))
    score, source = score_run(multi)
    assert source == 'continuous_only'
    assert 'macro_f1' not in score
    assert score['per_label.RR.roc_auc'] == 1


def test_multilabel_thresholds_masks_and_gallery_groups(make_run, tmp_path):
    run = a.load_run(make_run(task='rr_rh'))
    score, _ = score_run(run)
    assert score['subset_accuracy'] == .75
    assert score['hamming_loss'] == .125
    assert score['per_label.RR.f1'] == pytest.approx(2/3)
    assert score['per_label.RH.f1'] == 1
    assert score['micro_precision'] == 1
    frame = a.prediction_table([run])
    assert len(frame) == 8
    assert not frame[(frame.cluster_id == 'C') & (frame.label == 'RR')].correct.iloc[0]
    assert frame[(frame.cluster_id == 'C') & (frame.label == 'RH')].correct.iloc[0]
    for cluster in ['A','B','C','D']:
        np.save(tmp_path/f'{cluster}.npy', np.ones((8,8)))
    figs = a.plot_galleries(run, processed_root=tmp_path, radio_template='{cluster_id}.npy',
                           chandra_template='{cluster_id}_absent.npy')
    assert set(figs) == {'RR_correct','RR_incorrect','RH_correct','RH_incorrect'}
    plt.close('all')


def test_pairing_rejects_split_label_and_threshold_mismatch(make_run):
    left = a.load_run(make_run('left'))
    right = a.load_run(make_run('right', model='dual'))
    a.assert_comparable([left, right])
    assert len(a.compare_predictions([left, right])) == 8
    assert a.paired_comparisons([left,right]).paired.all()
    right.splits['samples']['t']['cluster_id'] = 'DIFFERENT'
    with pytest.raises(ValueError, match='partitions'):
        a.assert_comparable([left, right])
    assert not a.paired_comparisons([left,right]).paired.any()
    multi = a.load_run(make_run('multi', task='rr_rh'))
    with pytest.raises(ValueError, match='tasks'):
        a.assert_comparable([left,multi])
    other = a.load_run(make_run('other', task='rr_rh'))
    other.threshold_policy = {'method':'fixed','thresholds':{'RR':.5,'RH':.5}}
    with pytest.raises(ValueError, match='threshold'):
        a.assert_comparable([multi,other])


def test_config_fold_aggregation_missing_folds_and_repetitions(make_run):
    runs = [a.load_run(make_run(str(i), fold=i)) for i in [0,2]]
    repeated = a.load_run(make_run('repeat', fold=0))
    summary = a.aggregate_metrics(a.metrics_table(runs + [repeated]))
    assert len(summary) == 1
    assert summary.iloc[0].n_folds == 2
    assert summary.iloc[0].macro_f1_n == 2
    assert summary.iloc[0].n_runs == 3
    runs[1].config['learning_rate'] = .1
    assert runs[0].configuration_id != runs[1].configuration_id
    runs[0].status = 'running'
    assert a.aggregate_metrics(a.metrics_table([runs[0]])).empty


def test_fitted_normalization_bounds_aggregate_without_losing_policy(make_run):
    runs = [a.load_run(make_run(str(i), fold=i)) for i in range(3)]
    for i, run in enumerate(runs):
        run.config.update(
            modalities=['radio', 'legacy'],
            legacy={'normalization': {'scope': 'train', 'method': 'asinh_percentile',
                                      'lower_percentile': 30, 'upper_percentile': 99}},
            normalization_bounds={'legacy': [[.001 * (i + 1), .2 + .01 * i]]})
    original = json.dumps([run.config for run in runs], sort_keys=True)
    summary = a.aggregate_metrics(a.metrics_table(runs))
    assert len(summary) == 1
    assert summary.iloc[0].n_folds == 3
    assert summary.iloc[0].n_runs == 3
    assert json.dumps([run.config for run in runs], sort_keys=True) == original
    runs[1].config['legacy']['normalization']['upper_percentile'] = 95
    assert runs[0].configuration_id != runs[1].configuration_id
    runs[2].config['modalities'] = ['radio']
    assert runs[0].configuration_id != runs[2].configuration_id


def test_explicit_configuration_group_preserves_provenance(make_run):
    reference = a.load_run(make_run('reference', fold=0))
    path = make_run('other', fold=1)
    metadata = json.loads((path / 'config.json').read_text())
    metadata['config']['include_masks'] = True
    metadata['library_source_sha256'] = 'different-source'
    write(path / 'config.json', metadata)
    assert a.load_run(path).configuration_id != reference.configuration_id
    metadata['analysis_configuration_id'] = reference.configuration_id
    write(path / 'config.json', metadata)
    other = a.load_run(path)
    assert other.config['include_masks'] is True
    assert other.metadata['library_source_sha256'] == 'different-source'
    summary = a.aggregate_metrics(a.metrics_table([reference, other]))
    assert len(summary) == 1
    assert summary.iloc[0].n_folds == 2
    other.metadata['analysis_configuration_id'] = 'invalid'
    with pytest.raises(ValueError, match='analysis_configuration_id'):
        _ = other.configuration_id


def test_history_phases_variable_length(make_run):
    runs = [a.load_run(make_run(str(i), fold=i)) for i in [0,1]]
    runs[0].history = [dict(epoch=1, phase='frozen', phase_epoch=1, validation={'loss':2}),
                       dict(epoch=2, phase='tune', phase_epoch=1, validation={'loss':4})]
    runs[1].history = [dict(epoch=1, phase='frozen', phase_epoch=1, validation={'loss':4})]
    summary = a.aggregate_history(a.epoch_table(runs))
    assert summary.loc[summary.phase == 'frozen','mean'].iloc[0] == 3
    assert summary.loc[summary.phase == 'tune','count'].iloc[0] == 1
    assert np.isnan(summary.loc[summary.phase == 'tune','std'].iloc[0])
    a.plot_history(runs, aggregate=True)
    plt.close('all')


def test_missing_predictions_transfer_exports_and_filtering(make_run, tmp_path):
    path = make_run(transfer={'source_experiment':'source', 'phases':[{'name':'frozen','learning_rates':{'head':.01}}]})
    (path/'test_predictions.csv').unlink()
    run = a.load_run(path)
    assert not run.predictions
    assert score_run(run)[1] == 'stored'
    assert a.transfer_table([run]).iloc[0].phase == 'frozen'
    assert a.filter_runs([run], transfer_strategy='scratch') == []
    assert a.filter_runs([run], task_type=None) == [run]
    first = a.export_analysis([run], tmp_path/'exports')
    second = a.export_analysis([run], tmp_path/'exports')
    assert first != second
    assert (first/'run_registry.csv').exists()
    with pytest.raises(ValueError):
        a.export_analysis([run], path/'analysis')


def test_corrupt_predictions_and_manifest_leakage(make_run):
    path = make_run()
    frame = pd.read_csv(path/'test_predictions.csv')
    frame.loc[1,'cluster_id'] = 'A'
    frame.to_csv(path/'test_predictions.csv', index=False)
    run = a.load_run(path)
    assert 'test' not in run.predictions
    assert any('duplicate' in v for v in run.issues)
    d=json.loads((path/'splits.json').read_text()); d['train']=['a']
    write(path/'splits.json',d)
    assert a.load_run(path).status == 'invalid'


def test_notebook_executes_empty_root(tmp_path, monkeypatch):
    import nbformat
    from nbclient import NotebookClient
    path=Path(__file__).parents[1]/'notebooks/analyse_trainings.ipynb'
    nb=nbformat.read(path,as_version=4)
    monkeypatch.setenv('CLUSTERPREP_ANALYSIS_ROOT', str(tmp_path))
    # Execute identical cells, disable export in this empty-root regression.
    nb.cells[1].source += '\nEXPORT_TABLES = EXPORT_FIGURES = False'
    NotebookClient(nb, timeout=120, kernel_name='python3').execute(cwd=str(path.parent))


def test_cv_epoch_artifacts_and_oof_guards(make_run):
    path = make_run('cv/trial_0/fold_0', task='rr_rh')
    meta = json.loads((path/'config.json').read_text()); meta['execution']={'stage':'fold'}
    write(path/'config.json', meta)
    splits=json.loads((path/'splits.json').read_text())
    splits['validation'], splits['test'] = splits['test'], []
    write(path/'splits.json', splits)
    metrics=json.loads((path/'metrics.json').read_text());metrics['best_epoch']=2
    write(path/'metrics.json', metrics)
    frame=pd.read_csv(path/'test_predictions.csv')
    (path/'test_predictions.csv').unlink()
    rows=[]
    for r in frame.to_dict('records'):
        rows.append(dict(key=r['key'],cluster_id=r['cluster_id'],has_chandra=False,
            label=[r['label_RR'],r['label_RH']], label_mask=[1,1],
            probabilities=[r['probability_RR'],r['probability_RH']],
            prediction=[int(r['probability_RR']>=.7),int(r['probability_RH']>=.3)]))
    (path/'validation_epochs').mkdir()
    (path/'validation_epochs/epoch_2.json').write_text(json.dumps(rows))
    run=a.load_run(path)
    assert run.fold == 0 and 'validation' in run.predictions
    with pytest.raises(ValueError, match='incomplete'):
        a.out_of_fold_predictions([run])
    run.splits['train']=[]
    pooled=a.out_of_fold_predictions([run])
    assert pooled.prediction_role.eq('out_of_fold').all()
    with pytest.raises(ValueError, match='overlapping'):
        a.out_of_fold_predictions([run,run])
    run.metadata['execution']['stage']='final'
    with pytest.raises(ValueError, match='fixed test'):
        a.out_of_fold_predictions([run])


def test_partial_mask_and_absent_threshold_stored_metrics(make_run):
    path=make_run(task='rr_rh')
    frame=pd.read_csv(path/'test_predictions.csv')
    frame.loc[2,'known_RR']=False
    frame.loc[2,'label_RR']=np.nan
    frame.to_csv(path/'test_predictions.csv',index=False)
    run=a.load_run(path)
    assert score_run(run)[0]['per_label.RR.f1'] == 1
    assert score_run(run)[0]['hamming_loss'] == 0
    run.predictions={};run.thresholds=None
    run.metrics['test']={'macro_f1':.9,'roc_auc_macro':.8}
    score,_=score_run(run)
    assert 'macro_f1' not in score and score['roc_auc_macro']==.8


def test_protocol_separates_reference_labels_and_history(make_run):
    left=a.load_run(make_run('left'))
    right=a.load_run(make_run('right'))
    right.predictions['test'].loc[0,'label_DE']=0
    table=a.metrics_table([left,right])
    assert table.protocol_id.nunique()==2
    right.splits['samples']['t']['cluster_id']='OTHER'
    history=a.aggregate_history(a.epoch_table([left,right]))
    assert history.protocol_id.nunique()==2


def test_model_classification_noise_and_fold_views(make_run):
    runs = []
    for noise in ('gaussian_pixel', 'beam_correlated'):
        for model in ('fusion', 'dual_encoder_cnn'):
            for task in ('rr_binary', 'rh_binary'):
                for fold in range(5):
                    path = make_run(f'{noise}/{model}/{task}/{fold}', task=task, model=model, fold=fold)
                    meta = json.loads((path/'config.json').read_text())
                    meta['config']['augmentation'] = {'radio_noise': {'enabled': True, 'method': noise}}
                    write(path/'config.json', meta)
                    splits = json.loads((path/'splits.json').read_text())
                    splits['split_config']['n_folds'] = 5
                    write(path/'splits.json', splits)
                    runs.append(a.load_run(path))
    selected = a.filter_runs(runs, task='rr_binary', model='fusion', fold=[0, 2], noise_method='gaussian_pixel')
    assert len(selected) == 2
    table = a.metrics_table(runs)
    summary = a.aggregate_metrics(table)
    assert len(summary) == 8
    assert summary.n_folds.eq(5).all()
    assert set(summary.task) == {'rr_binary', 'rh_binary'}
    assert set(summary.noise_method) == {'gaussian_pixel', 'beam_correlated'}
    folds = a.metrics_by_fold(table)
    assert len(folds) == 40
    assert set(folds.fold) == set(range(5))
    figure = a.plot_performance(table, kind='heatmap')
    assert len(figure.axes) == 2
    assert all(ax.images[0].get_array().shape == (4, 5) for ax in figure.axes)
    assert all('binary' in ax.get_title() for ax in figure.axes)
    plt.close('all')


def test_folder_run_names_preserve_unique_identity(make_run):
    name = 'gaussian_pixel_dual_encoder_cnn_rr_binary_multimodal_fold_1'
    left = a.load_run(make_run(name + '-20261031T190114-6f1f9211', fold=1))
    right = a.load_run(make_run(name + '-20261101T190114-1234abcd', fold=1))
    assert left.run_name == right.run_name == name
    assert left.run_id != right.run_id
    assert left.configuration_name == name.removesuffix('_fold_1')
    # The config fixture deliberately has another name: the folder is authoritative.
    assert left.experiment_id != left.run_name
    table = a.display_table(a.registry_table([left, right]))
    assert list(table.run_name) == [name, name] and 'run_id' not in table
    assert len(a.metrics_table([left, right])) == 2
    summary = a.aggregate_metrics(a.metrics_table([left, right]))
    assert len(summary) == 1 and summary.iloc[0].n_runs == 2
    history = a.plot_history([left])
    assert any(name in label for label in history.axes[0].get_legend_handles_labels()[1])
    assert name in a.plot_diagnostics(left)._suptitle.get_text()
    pair = a.paired_comparisons([left, right])
    assert pair.left_name.iloc[0] == name
    assert 'left' not in a.display_table(pair)
    plain = a.load_run(make_run('custom_run_without_timestamp'))
    assert plain.run_name == 'custom_run_without_timestamp'
    plt.close('all')


def test_binary_diagnostics_single_target_percentages_and_named_export(make_run, tmp_path):
    run = a.load_run(make_run('binary_run-20261101T120000-1234abcd'))
    # DE is first in this fixture: select the target by meaning, not index 1.
    fig = a.plot_diagnostics(run)
    assert len(fig.axes) == 7 and fig.axes[0].get_title().startswith('DE:')
    assert [ax.get_title() for ax in fig.axes[-3:]] == ['Loss par epoch', 'Macro-F1 par epoch', 'Accuracy par epoch']
    np.testing.assert_array_equal(fig.axes[-3].lines[0].get_xdata(), [1, 2])
    np.testing.assert_array_equal(fig.axes[-3].lines[0].get_ydata(), [1, .5])
    np.testing.assert_array_equal(fig.axes[-3].lines[1].get_ydata(), [2, 1])
    assert fig.axes[0].images[0].get_cmap().name == 'Blues'
    np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), [[100, 0], [50, 50]])
    assert [t.get_text() for t in fig.axes[0].texts] == ['100.0% (2)', '0.0% (0)', '50.0% (1)', '50.0% (1)']
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ['NDE', 'DE']
    # Simulate the RH/non_RH schema, including both columns of its one-vs-rest table.
    run.config['task'] = 'rh_binary'
    run.metrics['class_names'] = ['non_RH', 'RH']
    run.predictions['test'] = run.predictions['test'].rename(columns={
        f'{prefix}_{old}': f'{prefix}_{new}' for prefix in ('label', 'prediction', 'known', 'probability')
        for old, new in [('DE', 'non_RH'), ('NDE', 'RH')]})
    rh = a.plot_diagnostics(run)
    assert len(rh.axes) == 7 and rh.axes[0].get_title().startswith('RH:')
    # Absent true classes have no invented rate.
    run.predictions['test']['label_RH'] = 0
    absent = a.plot_diagnostics(run)
    assert [t.get_text() for t in absent.axes[0].texts][2:] == ['— (0)', '— (0)']
    output = a.export_analysis([run], tmp_path/'exports', export_tables=False,
                              figures={f'confusion_matrices/{run.figure_stem}': rh})
    assert (output/'figures/confusion_matrices'/f'{run.figure_stem}.png').exists()
    assert run.run_name in run.figure_stem
    repeat = a.load_run(make_run('binary_run-20261102T120000-1234abcd'))
    assert run.run_name == repeat.run_name and run.figure_stem != repeat.figure_stem
    multi = a.load_run(make_run('multi_diagnostics', task='rr_rh'))
    assert len(a.plot_diagnostics(multi).axes) == 11
    plt.close('all')
