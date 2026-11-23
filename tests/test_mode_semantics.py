from copy import deepcopy
import pytest
import torch
from clusterprep.dataset import ClusterDataset
from clusterprep.models import DualEncoderCNN
from clusterprep.matched_runner import load_plan
from test_matched_dataset import catalog

@pytest.mark.parametrize('mode,expected',[('multimodal',2),('pair',1)])
@pytest.mark.parametrize('preprocessing',['historical','matched_resolution'])
@pytest.mark.parametrize('policy',['optional','required'])
def test_mode_alone_controls_chandra_population(catalog,mode,expected,preprocessing,policy):
    original,cid,root=catalog
    info=deepcopy(original)
    missing=deepcopy(info[cid]);missing['cluster_id']='absent';missing['chandra_path']=None
    missing['physical_record']['cluster_id']='absent';info['absent']=missing
    ds=ClusterDataset(info,list(info),['RR','RH'],mode=mode,modalities=['radio','xray'],
        task='rr_rh',missing_modality_policy=policy,radio_preprocessing=preprocessing,
        size=128,processed_dir=root)
    assert len(ds)==expected
    if mode=='multimodal':
        x=ds[1]
        assert not x['chandra'].any() and not x['chandra_mask'].any()
        assert torch.isfinite(x['raw']).all()
        model=DualEncoderCNN((3,128,128)).eval()
        with torch.no_grad():
            output=model(x['image'][None])
            altered=x['image'][None].clone();altered[:,1]=100
            assert torch.equal(output,model(altered))
        assert output.shape==(1,2) and torch.isfinite(output).all()

@pytest.mark.parametrize('mode,count',[('multimodal',154),('pair',65)])
def test_population_and_fixed_test_per_mode(mode,count):
    seen=[]
    for fold in range(5):
        a=load_plan('A',fold,mode=mode);b=load_plan('B',fold,mode=mode)
        assert a['partition']==b['partition']
        p=a['partition'];flat=p['train']+p['validation']+p['test']
        assert len(flat)==len(set(flat))==count
        assert len(p['test'])==13
        assert all(a['samples'][k]['chandra_path'] for k in flat) if mode=='pair' else True
        seen+=p['validation']
    assert len(seen)==len(set(seen))==count-13
