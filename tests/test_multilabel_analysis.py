"""Exact scores, validation-only thresholds, artifact compatibility and notebook."""
import csv
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from clusterprep.metrics import multilabel_metrics, resolve_thresholds, select_multilabel_thresholds
from clusterprep import multilabel_analysis as analysis
from clusterprep.training import ExperimentConfig, Trainer, run_experiment
from test_audit import catalog, single_thread


def controlled():
    return np.array([[0,0],[1,0],[0,1],[1,1]]), np.array([[.1,.2],[.8,.3],[.6,.7],[.9,.4]])


def test_exact_metrics_thresholds_and_brier():
    y, p = controlled()
    result = multilabel_metrics(y, p, thresholds={'RR':.7,'RH':.5})
    rr, rh = result['per_label']['RR'], result['per_label']['RH']
    assert rr['confusion_matrix'] == [[2,0],[0,2]]
    assert rh['confusion_matrix'] == [[2,0],[1,1]]
    assert [rh[k] for k in ['tn','fp','fn','tp']] == [2,0,1,1]
    assert rr['roc_auc'] == rh['roc_auc'] == rr['average_precision'] == rh['average_precision'] == 1
    assert rr['n_known'] == 4 and rr['n_positive'] == rr['n_negative'] == 2
    assert rr['prevalence'] == .5 and rr['balanced_accuracy'] == 1
    assert rh['balanced_accuracy'] == .75
    assert result['balanced_accuracy'] == .875
    assert result['macro_f1'] == pytest.approx(5/6)
    assert result['micro_f1'] == pytest.approx(6/7)
    assert result['hamming_accuracy'] == .875 and result['hamming_loss'] == .125
    assert result['subset_accuracy'] == .75
    assert rr['brier'] == pytest.approx(.105)
    assert rh['brier'] == pytest.approx(.145)
    changed = multilabel_metrics(y, p, thresholds={'RR':.4,'RH':.3})
    for metric in ['roc_auc_macro','average_precision_macro','brier_macro']:
        assert result[metric] == changed[metric]
    assert result['macro_f1'] != changed['macro_f1']
    json.dumps(result, allow_nan=False)


def test_unknown_values_and_aggregation_counts():
    y, p = controlled()
    y = y.astype(float)
    mask = np.ones_like(y)
    mask[:,1] = 0
    y[:,1] = np.nan
    result = multilabel_metrics(y,p,target_mask=mask)
    assert result['roc_auc_macro_n_labels'] == result['macro_f1_n_labels'] == 1
    assert result['per_label']['RH']['n_known'] == 0
    assert result['per_label']['RH']['precision'] is None
    assert result['subset_accuracy'] is None and result['n_fully_known'] == 0
    assert result['n_known'] == 4
    json.dumps(result, allow_nan=False)
    assert multilabel_metrics([[1,0],[1,0]],[[.9,.1],[.8,.2]])['roc_auc_macro_n_labels'] == 0
    with pytest.raises(ValueError, match='connus'):
        multilabel_metrics([[np.nan,0]],[[.5,.5]])
    with pytest.raises(ValueError):
        multilabel_metrics(y,p,mask=mask,target_mask=mask)


@pytest.mark.parametrize('thresholds', [{'RR':.5}, {'RR':np.nan,'RH':.5}, [0,.5], [.5,1], [True,.5], [.5]])
def test_bad_thresholds(thresholds):
    with pytest.raises(ValueError):
        resolve_thresholds(thresholds)


def test_selection_validation_only_ties_and_fallback():
    y,p=controlled()
    selected = select_multilabel_thresholds(y,p,grid=[.35,.5,.7])
    assert selected['thresholds'] == {'RR':.7,'RH':.35}
    assert selected == select_multilabel_thresholds(y,p,grid=[.7,.35,.5])
    with pytest.raises(ValueError, match='validation uniquement'):
        select_multilabel_thresholds(y,p,split='test')
    perfect = np.where(y,.9,.1)
    assert select_multilabel_thresholds(y,perfect,grid=[.4,.6])['thresholds'] == {'RR':.6,'RH':.6}
    degenerate = select_multilabel_thresholds(np.ones((3,2)),np.ones((3,2))*.7)
    assert degenerate['thresholds'] == {'RR':.5,'RH':.5}
    assert degenerate['per_label']['RR']['status'] == 'single_class_or_no_known_labels'
    impossible = select_multilabel_thresholds(y,np.ones_like(p)*.2,method='recall_at_precision',min_precision=1,grid=[.1])
    assert impossible['per_label']['RR']['status'] == 'precision_constraint_unattainable'
    attained = select_multilabel_thresholds(y,p,method='recall_at_precision',min_precision=1,grid=[.35,.7])
    assert attained['thresholds'] == {'RR':.7,'RH':.35}


@pytest.fixture
def demo(tmp_path):
    return analysis.create_demo_runs(tmp_path/'demo')


def test_artifact_loader_tables_bootstrap_and_comparison(demo, tmp_path):
    runs = [analysis.load_run(p) for p in demo]
    assert analysis.assert_comparable(runs)
    assert len(analysis.discover_runs(tmp_path)) == 3
    run=runs[0]
    rows=analysis.annotation_rows(run)
    assert len(rows) == 48
    assert sum(r['combination']=='unknown' for r in rows) > 0
    assert any(r['candidate_RR'] for r in rows)
    assert len(analysis.annotation_summary(run)) == 6
    p=run.predictions['test']
    assert not np.allclose(p.probabilities.sum(1),1)
    ci=analysis.bootstrap_clusters(p,iterations=12,seed=2)
    assert ci == analysis.bootstrap_clusters(p,iterations=12,seed=2)
    assert all(r['n_valid']<=12 for r in ci)
    with pytest.raises(ValueError,match='clusters uniques'):
        analysis.bootstrap_clusters(p.subset(np.array([0,0])),iterations=3)
    assert analysis.validation_selection(run)['selection_split']=='validation'
    assert len(analysis.threshold_curves(run))==38
    output=analysis.create_output_directory(tmp_path/'exports')
    analysis.export_rows(output/'scores.csv',[analysis.flatten_metrics(p.scores())])
    with pytest.raises(FileExistsError):
        analysis.export_rows(output/'scores.csv',[])
    json.dumps(p.scores(),allow_nan=False)
    runs[1].config['candidate_policy']='positive'
    with pytest.raises(ValueError,match='incompatibles'):
        analysis.assert_comparable(runs)


def rewrite_csv(path, change):
    with path.open() as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
    fields, rows = change(fields,rows)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


@pytest.mark.parametrize('corruption',['duplicate','identity','leak','probability','policy','predicted','test_keys'])
def test_loader_rejects_corruption(demo,corruption):
    path=demo[0]
    if corruption in {'duplicate','identity','probability','predicted'}:
        def change(fields,rows):
            if corruption=='duplicate': rows[1]['cluster_id']=rows[0]['cluster_id']
            if corruption=='identity': rows[0]['cluster_id']='other'
            if corruption=='probability': rows[0]['probability_RR']='1.2'
            if corruption=='predicted': rows[0]['prediction_RR']=str(1-int(rows[0]['prediction_RR']))
            return fields,rows
        rewrite_csv(path/'test_predictions.csv',change)
    elif corruption=='policy':
        value=json.loads((path/'metrics.json').read_text());value['config']['candidate_policy']='all'
        (path/'metrics.json').write_text(json.dumps(value))
    else:
        value=json.loads((path/'splits.json').read_text())
        if corruption=='leak': value['train'][0]=value['test'][0]
        else: value['test']=value['test'][:-1]
        (path/'splits.json').write_text(json.dumps(value))
    with pytest.raises(ValueError): analysis.load_run(path)


def test_legacy_schema_explicit_masks_and_history(demo):
    path=demo[0]
    for split in ['validation','test']:
        def change(fields,rows):
            names=[f for f in fields if not f.startswith('known_')]
            return names,[{k:v for k,v in r.items() if k in names} for r in rows]
        rewrite_csv(path/f'{split}_predictions.csv',change)
    (path/'history.jsonl').unlink()
    value=json.loads((path/'metrics.json').read_text());value['class_names']=['DE','NDE']
    (path/'metrics.json').write_text(json.dumps(value))
    run=analysis.load_run(path)
    assert any('ancien CSV' in s for s in run.issues)
    assert any('DE/NDE erroné' in s for s in run.issues)
    assert any('history.jsonl absent' in s for s in run.issues)
    assert not run.predictions['test'].mask.all()


def test_fold_summary_rejects_overlapping_tests(demo):
    runs=[analysis.load_run(p) for p in demo[:2]]
    runs[1].config=dict(runs[0].config,name='other')
    for i,run in enumerate(runs):
        run.splits['split_config']={'n_folds':2,'fold_index':i,'split_seed':42}
    with pytest.raises(ValueError,match='chevauchent'):
        analysis.fold_summary(runs)
    runs[1].predictions['test'].cluster_ids=np.array(['fold2-'+v for v in runs[1].predictions['test'].cluster_ids])
    summary=analysis.fold_summary(runs)
    assert len(summary)==4 and all(r['n_folds_present']==2 for r in summary)


def test_threshold_training_integration_and_single_test_evaluation(catalog,tmp_path):
    config=ExperimentConfig('thresholds',task='rr_rh',candidate_policy='ignore',threshold_selection='f1',
        threshold_grid=[.25,.5,.75],epochs=2,batch_size=4,size=16,width=2,device='cpu',
        augment=False,use_mixup=False,processed_dir=None,verbose=False)
    splits=([f'c{i}.fits' for i in range(12)], [f'c{i}.fits' for i in range(12,18)], [f'c{i}.fits' for i in range(18,24)])
    calls=[]
    original=Trainer.evaluate
    def evaluate(self,*args,**kwargs):
        calls.append(kwargs.get('description'))
        return original(self,*args,**kwargs)
    with patch.object(Trainer,'evaluate',evaluate):
        path=run_experiment(catalog,config,tmp_path/'runs',splits=splits)
    assert calls.count('Test final')==1
    run=analysis.load_run(path)
    selected=analysis.validation_selection(run,method='f1',grid=config.threshold_grid)
    assert selected['thresholds']==run.thresholds
    assert json.loads((path/'thresholds.json').read_text())['selection_split']=='validation'
    ckpt=torch.load(path/'best.pt',weights_only=True)
    assert ckpt['evaluation']['thresholds']==run.thresholds
    actual=run.predictions['test'].scores(run.thresholds)
    assert actual['macro_f1']==run.metrics['test']['macro_f1']
    assert actual['per_label']['RR']['tp']==run.metrics['test']['per_label']['RR']['tp']
    # Changing test truths/scores cannot alter a validation-only selection.
    run.predictions['test'].targets[:]=1-run.predictions['test'].targets
    run.predictions['test'].probabilities[:]=0
    assert analysis.validation_selection(run,method='f1',grid=config.threshold_grid)['thresholds']==selected['thresholds']
    with pytest.raises(ValueError,match='réservée'):
        replace(config,task='de_nde')


def test_masked_nan_targets_training():
    from torch.utils.data import DataLoader
    config=ExperimentConfig('nan-mask',task='rr_rh',use_class_weights=False,use_mixup=False,use_scheduler=False)
    model=torch.nn.Linear(2,2)
    trainer=Trainer(model,config,['RR','RH'],[1,1],steps_per_epoch=1,input_adapter=lambda b:b['raw'])
    data=[{'raw':torch.ones(2),'label':torch.tensor([float('nan'),1.]),'label_mask':torch.tensor([0.,1.])},
          {'raw':torch.zeros(2),'label':torch.tensor([1.,0.]),'label_mask':torch.ones(2)}]
    metrics,_=trainer.train_one_epoch(DataLoader(data,batch_size=2))
    assert np.isfinite(metrics['loss'])


def test_notebook_smoke(tmp_path, monkeypatch):
    nbformat=pytest.importorskip('nbformat')
    pytest.importorskip('pandas')
    from IPython.core.interactiveshell import InteractiveShell
    notebook=nbformat.read(Path(__file__).parents[1]/'notebooks'/'analyse_multilabel.ipynb',as_version=4)
    monkeypatch.setenv('CLUSTERPREP_ANALYSIS_OUTPUT_ROOT',str(tmp_path/'exports'))
    monkeypatch.setenv('CLUSTERPREP_BOOTSTRAP_ITERATIONS','4')
    monkeypatch.setenv('CLUSTERPREP_THRESHOLD_METHOD','f1')
    monkeypatch.delenv('CLUSTERPREP_ANALYSIS_RUN_DIRS',raising=False)
    # Execute every code cell with an actual IPython shell, without requiring a
    # network-bound kernel. A separate nbclient run is used for saved outputs.
    shell=InteractiveShell()
    for cell in notebook.cells:
        if cell.cell_type=='code':
            result=shell.run_cell(cell.source)
            result.raise_error()
            # Isolate the smoke test from user-edited local notebook paths.
            if 'RUN_DIRS =' in cell.source and 'RESULTS_ROOT =' in cell.source:
                shell.user_ns['RUN_DIRS'] = []
                shell.user_ns['RESULTS_ROOT'] = None
    outputs=list((tmp_path/'exports').glob('analysis-*'))
    assert len(outputs)==1
    assert (outputs[0]/'test_metrics.json').exists()
    assert len(list(outputs[0].glob('*.png')))>=16
    assert json.loads((outputs[0]/'analysis_config.json').read_text())['synthetic'] is True


def test_prediction_order_does_not_change_cohort_and_truth_mismatch_rejected(demo):
    original=analysis.load_run(demo[0])
    for split in ['validation','test']:
        rewrite_csv(demo[1]/f'{split}_predictions.csv',lambda fields,rows:(fields,list(reversed(rows))))
    shuffled=analysis.load_run(demo[1])
    assert analysis.assert_comparable([original,shuffled])
    i,j=np.argwhere(shuffled.predictions['test'].mask)[0]
    shuffled.predictions['test'].targets[i,j]=1-shuffled.predictions['test'].targets[i,j]
    with pytest.raises(ValueError,match='incompatibles'):
        analysis.assert_comparable([original,shuffled])


def test_threshold_wizard_json(tmp_path):
    from clusterprep.config_wizard import create_config
    def answer(prompt):
        if prompt.startswith('Tâche'):return 'rr_rh'
        if prompt.startswith('Seuils RR/RH'):return '{"RR":0.4,"RH":0.65}'
        if prompt.startswith('Sélection des seuils'):return 'f1'
        if prompt.startswith('Grille de seuils'):return '[0.2,0.5,0.8]'
        return ''
    path=create_config(tmp_path/'config.json',input_fn=answer,print_fn=lambda *args:None)
    config=json.loads(path.read_text())['experiments'][0]
    assert config['thresholds']=={'RR':.4,'RH':.65}
    assert config['threshold_selection']=='f1'
    assert config['threshold_grid']==[.2,.5,.8]


def test_bad_json_schema_is_readable(demo):
    (demo[0]/'metrics.json').write_text('[]')
    with pytest.raises(ValueError,match='Objet JSON requis'):
        analysis.load_run(demo[0])
