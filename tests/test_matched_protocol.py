"""Leakage, masked supervision and phase-6 artifact contract."""
import json
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from clusterprep.matched_protocol import prepare_folds, calibrate, export_oof, paired_analysis
from clusterprep.matched_cohort import cohort_annotation
from clusterprep.training import ExperimentConfig, Trainer, compute_task_counts
from clusterprep.metrics import multilabel_metrics


def population():
    labels=[['NDE']]*21+[['RR']]*4+[['RH']]*32+[['RR','RH']]*8
    return {str(i):dict(cluster_id=str(i),**cohort_annotation(v)[0]) for i,v in enumerate(labels)}


def test_folds_partition_masks_and_reproducibility():
    info=population(); folds=prepare_folds(info)
    assert folds==prepare_folds(info)
    assert sorted(k for f in folds for k in f['test'])==sorted(info)
    for f in folds:
        assert [len(f[n]) for n in ('train','validation','test')]==[39,13,13]
        for n in ('train','validation','test'):
            assert f['distributions'][n]['RR_positive']>0
            assert f['distributions'][n]['RR_negative']>0
    assert sum(f['distributions']['test']['combination_1?']==0 for f in folds)==1


def rows(ids):
    return [dict(cluster_id=k,label=[i%2,(i+1)%2],label_mask=[1,1],probabilities=[.2+.6*(i%2),.8-.6*(i%2)]) for i,k in enumerate(ids)]


def test_calibration_and_oof_refuse_test_leakage(tmp_path):
    partition=dict(train=['a','b'],validation=['c','d'],test=['e','f'])
    selection=calibrate(rows(partition['validation']),partition)
    with pytest.raises(ValueError): calibrate(rows(partition['test']),partition)
    with pytest.raises(ValueError): calibrate(rows(['c','c']),partition)
    result=export_oof(rows(partition['test']),partition,selection,'RAW',42,tmp_path/'oof.csv')
    assert [r['cluster_id'] for r in result]==['e','f']
    with pytest.raises(ValueError): export_oof(rows(['c','d']),partition,selection,'RAW',42,tmp_path/'bad.csv')


def test_paired_bootstrap_same_products_zero_difference():
    data=rows(['a','b','c','d']); data[0]['label_mask']=[0,1]
    r=paired_analysis(data,list(reversed(data)),{'RR':.5,'RH':.5},{'RR':.5,'RH':.5},draws=20)
    assert r['metrics_RAW']['per_label']['RR']['n_known']==3
    for interval in r['intervals'].values():
        assert interval['delta_T50_minus_RAW']['ci95']==[0.,0.]
    bad=deepcopy(data); bad[0]['label_mask']=[1,1]
    with pytest.raises(ValueError): paired_analysis(data,bad,{}, {},draws=2)


def trainer(model):
    config=ExperimentConfig(name='test',task='rr_rh',epochs=1,use_scheduler=False,
        use_compile=False,use_mixup=False,verbose=False,weight_decay=.1)
    return Trainer(model,config,('RR','RH'),[1,1],n_train=[2,2],steps_per_epoch=2,
                   input_adapter=lambda b:b['raw'])


def test_masked_loss_matches_known_elements_and_gradients():
    model=torch.nn.Linear(2,2,bias=False); t=trainer(model)
    x=torch.eye(2); y=torch.tensor([[1.,float('nan')],[0.,1.]])
    mask=torch.tensor([[1.,0.],[1.,1.]])
    data=[dict(raw=x[i],label=y[i],label_mask=mask[i]) for i in range(2)]
    logits=model(x); clean=torch.nan_to_num(y)
    expected=(torch.nn.functional.binary_cross_entropy_with_logits(logits,clean,reduction='none')*mask).sum()/3
    metrics,_=t.evaluate(DataLoader(data,batch_size=2))
    assert metrics['loss']==pytest.approx(expected.item())
    t.train_one_epoch(DataLoader(data,batch_size=2))
    assert model.weight.grad[1,0]==0
    assert model.weight.grad[0,0]!=0
    assert metrics['per_label']['RH']['n_known']==1


def test_unknown_batch_skips_optimizer_and_all_unknown_split_rejected():
    model=torch.nn.Linear(2,2); t=trainer(model)
    before={k:v.clone() for k,v in model.state_dict().items()}
    unknown=[dict(raw=torch.ones(2),label=torch.tensor([float('nan'),-1.]),label_mask=torch.zeros(2))]
    with pytest.raises(ValueError,match='Aucun label connu'):
        t.train_one_epoch(DataLoader(unknown,batch_size=1))
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert not t.optimizer.state
    mixed=unknown+[dict(raw=torch.ones(2),label=torch.ones(2),label_mask=torch.ones(2))]
    result,_=t.train_one_epoch(DataLoader(mixed,batch_size=1))
    assert result['n_known']==2
    assert all(v['step']==1 for v in t.optimizer.state.values())


def test_counts_exclude_masked_positives_and_holdout():
    info={'train':dict(labels=['RR','RH'],rr_rh_known_mask=[0,1]),
          'test':dict(labels=['RR','RH'])}
    assert compute_task_counts(info,['train'],'rr_rh')==[0,1]
    info['test']['labels']=['NDE']
    assert compute_task_counts(info,['train'],'rr_rh')==[0,1]


def test_unknown_targets_never_change_metrics():
    y=np.array([[1.,np.nan],[0.,1.]])
    p=np.array([[.9,.8],[.1,.6]]); mask=np.array([[1,0],[1,1]])
    a=multilabel_metrics(y,p,target_mask=mask)
    y[0,1]=1
    assert a==multilabel_metrics(y,p,target_mask=mask)
    assert a['per_label']['RH']['roc_auc'] is None


def test_masked_mixup_keeps_unknown_gradients_zero(monkeypatch):
    model=torch.nn.Linear(2,2,bias=False); t=trainer(model)
    t.config=replace(t.config,use_mixup=True,mixup_probability=1.)
    monkeypatch.setattr('numpy.random.beta',lambda *a:.25)
    monkeypatch.setattr('torch.randperm',lambda n,device=None:torch.tensor([1,0],device=device))
    data=[dict(raw=torch.tensor([1.,0.]),label=torch.tensor([1.,float('nan')]),label_mask=torch.tensor([1.,0.])),
          dict(raw=torch.tensor([0.,1.]),label=torch.tensor([0.,-1.]),label_mask=torch.tensor([1.,0.]))]
    result,_=t.train_one_epoch(DataLoader(data,batch_size=2))
    assert torch.equal(model.weight.grad[1],torch.zeros(2))
    assert result['per_label']['RH']['n_known']==0
    assert result['per_label']['RH']['f1'] is None


def test_strata_keep_empty_bins_and_denominators():
    from clusterprep.matched_protocol import stratified_comparisons
    data=rows(['a','b']); meta={'a':dict(coverage_CHANDRA=.05,fwhm_over_r500=.09),
                             'b':dict(coverage_CHANDRA=.7,fwhm_over_r500=.13)}
    result=stratified_comparisons(data,data,{'RR':.5,'RH':.5},{'RR':.5,'RH':.5},meta,draws=2)
    assert sum(r['n_clusters'] for r in result if r['variable']=='coverage_CHANDRA')==2
    assert any(r['comparison'] is None for r in result)


def test_oof_metrics_keep_per_fold_cutoffs():
    from clusterprep.matched_protocol import compare_oof_records
    data=[]
    for i in range(4):
        r=dict(cluster_id=str(i),fold=i%2,seed=42)
        for name in ('RR','RH'):
            r.update({name+'_target':i%2,name+'_known':1,name+'_probability':.4 if i%2 else .6,
                      name+'_threshold':.3 if i%2 else .7})
        data.append(r)
    result=compare_oof_records(data,data,draws=10)
    assert result['metrics_RAW']['macro_f1']==1
    assert result['metrics_RAW']['per_label']['RR']['roc_auc']==0
    assert result['metrics_RAW']['per_label']['RR']['precision_denominator']==2
    bad=deepcopy(data); bad[0]['seed']=123
    with pytest.raises(ValueError): compare_oof_records(data,bad,draws=2)
