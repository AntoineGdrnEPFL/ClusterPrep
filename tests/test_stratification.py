"""Scientific RR/RH strata and legacy DE/NDE regression tests."""
import json
from collections import Counter
from unittest.mock import patch
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from clusterprep.labels import task_target
from clusterprep.dataset import ClusterDataset
from clusterprep.splits import eligible_for_task, multilabel_stratum, split_data_keys, split_distribution
from clusterprep.training import ExperimentConfig, run_experiments


def inventory(n=20):
    return {f'{s}_{i}': {'cluster_id': f'{s}_{i}', 'labels': set(labels),
                        'chandra_path': 'x' if i % 2 else None}
            for s, labels in [('00', ['NDE']), ('10', ['DE','RR']), ('01', ['DE','RH']), ('11', ['DE','RR','RH'])]
            for i in range(n)}


@pytest.mark.parametrize('labels,target,stratum', [(['NDE'],[0,0],'00'), (['RR'],[1,0],'10'), (['RH'],[0,1],'01'), (['RR','RH'],[1,1],'11'), (['cRR'],[1,0],'10')])
def test_targets(labels, target, stratum):
    item = {'labels': labels, 'cluster_id': 'a'}
    assert eligible_for_task(item, 'rr_rh')
    assert multilabel_stratum(item) == stratum
    ds = ClusterDataset({'a': item}, ['a'], ['RR','RH'], task='rr_rh')
    assert ds._make_target(item).tolist() == target


def test_unknown_and_contradictions():
    assert not eligible_for_task({'labels':['DE']}, 'rr_rh')
    with pytest.raises(ValueError, match='éligible'):
        multilabel_stratum({'labels':['DE']})
    assert multilabel_stratum({'labels':['cRR']}, 'ignore') == '?0'
    assert multilabel_stratum({'labels':['cRR','cRH']}, 'ignore') == '??'
    for labels in [['NDE','RR'], ['NDE','RH'], ['NDE','cRR'], ['NDE','DE']]:
        with pytest.raises(ValueError, match='contradictoires'):
            eligible_for_task({'labels':labels}, 'rr_rh')


@pytest.mark.parametrize('folds', [None, 5])
def test_four_strata_and_groups(folds):
    info = inventory()
    info['alias'] = {**info['11_0'], 'labels': {'RH'}}
    info['11_0']['labels'] = {'RR'}
    info['excluded'] = {'cluster_id':'excluded','labels':{'DE'}}
    options = dict(task='rr_rh', n_folds=folds, seed=17)
    splits = split_data_keys(info, **options)
    assert splits == split_data_keys(info, **options)
    ids = [{info[k]['cluster_id'] for k in split} for split in splits]
    assert not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    assert len(set.union(*ids)) == 80
    dist = split_distribution(info, splits, 'rr_rh')
    for name, count in [('train',12),('validation',4),('test',4)]:
        assert dist[name] == dict.fromkeys(['00','10','01','11'],count)
    tests = [set(split_data_keys(info, task='rr_rh', n_folds=5, fold_index=i)[2]) for i in range(5)]
    assert len(set.union(*tests)) == 81
    assert sum(map(len,tests)) == 81


def test_legacy_exact():
    info = {str(i): {'cluster_id':str(i),'class':'DE' if i%2 else 'NDE'} for i in range(100)}
    ids = sorted(info)
    y = [info[k]['class'] for k in ids]
    tv, test = train_test_split(ids,test_size=.2,random_state=42,stratify=y)
    train,val = train_test_split(tv,test_size=.25,random_state=42,stratify=[info[k]['class'] for k in tv])
    assert split_data_keys(info) == (train,val,test)


@pytest.mark.parametrize('folds',[None,5])
def test_rare_iterative(folds):
    info=inventory(12)
    info={k:v for k,v in info.items() if not k.startswith('11_') or k=='11_0'}
    with pytest.warns(UserWarning,match='itérative'):
        splits=split_data_keys(info,task='rr_rh',n_folds=folds)
    with pytest.warns(UserWarning):
        assert splits==split_data_keys(info,task='rr_rh',n_folds=folds)
    assert sum(map(len,splits))==len(info)
    with patch.dict('sys.modules', {'iterstrat.ml_stratifiers':None}):
        with pytest.raises(ValueError,match='effectifs'):
            split_data_keys(info,task='rr_rh',n_folds=folds)


def test_diagnostics_masks():
    info={'a':{'cluster_id':'a','labels':{'cRR'}},'b':{'cluster_id':'b','labels':{'NDE'}}}
    with pytest.warns(UserWarning,match='ROC-AUC'):
        d=split_distribution(info, (['a','b'],['b'],['a']), 'rr_rh','ignore')
    assert d['train']['00']==1 and d['train']['?0']==1
    assert d['labels']['train']['RR']==dict(positive=0,negative=1,unknown=1,roc_auc_defined=False)


def test_suite_task_and_population(tmp_path):
    info=inventory(30)
    configs=[ExperimentConfig(str(i),task=task,mode=mode) for i,(task,mode) in enumerate([
        ('de_nde','raw'),('rr_rh','raw'),('rr_rh','multimodal'),('rr_rh','pair')])]
    captured=[]
    def capture(info, config, *args, **kwargs):
        captured.append(kwargs['splits'])
    with patch('clusterprep.training.run_experiment',side_effect=capture):
        run_experiments(info,configs,tmp_path)
    assert captured[1]==captured[2]
    assert captured[0]!=captured[1]
    assert all(info[k]['chandra_path'] for split in captured[3] for k in split)
    assert sum(map(len,captured[3]))==60


def test_too_small_and_conflicting_aliases():
    info=inventory(1)
    with pytest.raises(ValueError,match='effectifs'):
        split_data_keys(dict(list(info.items())[:2]),task='rr_rh')


def test_generated_suite_deduplicates_aliases(tmp_path):
    info=inventory()
    info['alias']={**info['11_0'], 'labels':{'RH'}}
    info['11_0']['labels']={'RR'}
    config=ExperimentConfig('aliases',task='rr_rh')
    with patch('clusterprep.training.run_experiment') as runner:
        run_experiments(info,[config],tmp_path)
    splits=runner.call_args.kwargs['splits']
    assert sum(map(len,splits))==80
    ds=ClusterDataset(info,['11_0'],['RR','RH'],task='rr_rh')
    assert ds._make_target(ds.data_info['11_0']).tolist()==[1,1]
    info['alias']['labels']={'NDE'}
    with pytest.raises(ValueError,match='contradictoires'):
        split_data_keys(info,task='rr_rh')
