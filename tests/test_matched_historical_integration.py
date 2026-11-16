"""Exercise historical setup/exports with fit/evaluate stubs: no training."""
from copy import deepcopy
from dataclasses import replace
import json
import torch
from clusterprep import training,matched_runner
from clusterprep.matched_dataset import MatchedDatasetAdapter
from test_matched_dataset import catalog


def test_historical_setup_train_normalization_and_no_test_access(catalog,tmp_path,monkeypatch):
    source,cid,cache=catalog
    info={}
    for name,labels in [('positive',{'RR','RH'}),('negative',{'NDE'}),('validation',{'RR'}),('external',{'RH'})]:
        item=deepcopy(source[cid]);item['cluster_id']=name
        item['physical_record']['cluster_id']=name;item['labels']=labels
        item['rr_rh_known_mask']=[1,0] if name=='validation' else [1,1]
        info[name]=item
    seen=[];load=MatchedDatasetAdapter.load
    def guarded(self,item,*args,**kwargs):
        assert item['cluster_id']!='external'
        seen.append(item['cluster_id']);return load(self,item,*args,**kwargs)
    monkeypatch.setattr(MatchedDatasetAdapter,'load',guarded)
    config=replace(matched_runner.training_configuration(matched_runner.load_plan('A'),'cpu',200),
                   matched_cache_dir=str(cache),processed_dir=str(cache),verbose=False,use_compile=False)
    constructors=[];original=training.ClusterDataset
    def dataset(*a,**kw):
        ds=original(*a,**kw);constructors.append(ds);return ds
    monkeypatch.setattr(training,'ClusterDataset',dataset)
    def fit(self,train,val,run_dir=None,**kwargs):
        assert len(train.dataset)==48 and len(val.dataset)==1
        assert train.drop_last and len(train)==3
        assert self.steps_per_epoch==3
        assert self.model.__class__.__name__=='DualEncoderCNN'
        assert train.dataset._matched is not None and val.dataset._matched is not None
        assert train.dataset.normalization_bounds==val.dataset.normalization_bounds
        assert val.dataset[0]['label_mask'].tolist()==[1,0]
        assert set(seen)<= {'positive','negative','validation'}
        assert self.criterion.pos_weight.tolist()==[1.,1.]
        self.best_epoch=1;self.stopping.best=.2
        torch.save({},run_dir/'best.pt')
        return [{'epoch':1}]
    evaluated=[]
    def evaluate(self,loader,**kwargs):
        evaluated.append(list(loader.dataset.keys))
        assert loader.dataset.keys==['validation']
        return {'loss':.2},[dict(cluster_id='validation',key='validation',has_chandra=True,
                  label=[1,0],label_mask=[1,0],probabilities=[.6,.4],prediction=[1,0],loss=.2)]
    monkeypatch.setattr(training.Trainer,'fit',fit)
    monkeypatch.setattr(training.Trainer,'evaluate',evaluate)
    result=training.run_experiment(info,config,tmp_path/'runs',class_names=('RR','RH'),
                  splits=(['positive','negative'],['validation'],['external']))
    assert evaluated==[['validation']]
    assert all('external' not in ds.keys for ds in constructors)
    fitting=constructors[0]
    assert fitting.keys==['positive','negative'] and not fitting.augment
    assert json.loads((result/'status.json').read_text())['test_evaluated'] is False
    splits=json.loads((result/'splits.json').read_text())
    assert splits['reserved_external_test']==['external'] and splits['test']==[]
    assert not (result/'test_predictions.csv').exists()
    assert json.loads((result/'thresholds.json').read_text())['thresholds']=={'RR':.5,'RH':.5}
