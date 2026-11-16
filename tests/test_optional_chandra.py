"""Optional CHANDRA extension, using the existing DualEncoderCNN."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from astropy.io import fits
from test_matched_dataset import catalog
from test_matched_protocol import population
from clusterprep.matched_cohort import make_dataset
from clusterprep.matched_protocol import prepare_folds
from clusterprep.optional_chandra_cohort import extend_folds, configuration
from clusterprep.modalities import fit_normalization
from clusterprep.models import DualEncoderCNN


def test_missing_zero_pair_cache_and_presence_identity(catalog):
    info,cid,root=catalog
    present=deepcopy(info)
    info[cid]['chandra_path']=None
    a=make_dataset(info,[cid],'RAW',root); b=make_dataset(info,[cid],'T50kpcSUB',root)
    x,y=a[0],b[0]
    assert not x['has_chandra']
    assert x['resolution_metadata']['chandra_status']=='absent'
    assert torch.count_nonzero(x['chandra'])==torch.count_nonzero(x['chandra_mask'])==0
    assert torch.equal(x['chandra'],y['chandra']) and torch.equal(x['raw_mask'],y['raw_mask'])
    assert a.get_resolution_metadata(0)['cache_key']==b.get_resolution_metadata(0)['cache_key']
    # A new store must read the on-disk zero/mask pair, not rebuild it.
    fresh=make_dataset(info,[cid],'RAW',root)
    assert fresh.get_resolution_metadata(0)['cache']['status']=='reused'
    physical=fresh._modality_images(fresh.data_info[cid])
    assert np.all(physical['chandra']==0) and not physical['chandra_mask'].any()
    covered=make_dataset(present,[cid],'RAW',root)
    assert covered.get_resolution_metadata(0)['cache_key']!=a.get_resolution_metadata(0)['cache_key']
    assert covered[0]['has_chandra']
    assert 0<covered.get_resolution_metadata(0)['coverage']['CHANDRA']<1


def test_present_without_coverage_is_not_absent(catalog,tmp_path):
    info,cid,root=catalog
    path=tmp_path/'uncovered.fits'
    fits.writeto(path,np.full((256,256),np.nan),fits.getheader(info[cid]['chandra_path']))
    info[cid]['chandra_path']=str(path)
    ds=make_dataset(info,[cid],'RAW',root)
    assert ds[0]['has_chandra']
    assert ds[0]['resolution_metadata']['chandra_status']=='present_uncovered'
    assert not ds[0]['chandra_mask'].any()


def test_present_invalid_file_not_silently_missing(catalog,tmp_path):
    info,cid,root=catalog
    path=tmp_path/'broken.fits';path.write_text('not FITS')
    info[cid]['chandra_path']=str(path)
    with pytest.raises((OSError,ValueError)):
        make_dataset(info,[cid],'RAW',root)[0]


def test_global_normalization_ignores_missing_xray(catalog):
    info,cid,root=catalog
    extra=deepcopy(info[cid]);extra.update(cluster_id='absent',chandra_path=None)
    extra['physical_record']['cluster_id']='absent'
    info['absent']=extra
    options={m:dict(scope='train',method='asinh_percentile',lower_percentile=30,upper_percentile=99,alpha=10.) for m in ('radio','xray')}
    original=fit_normalization(make_dataset(info,[cid],'RAW',root,normalization=options))
    extended=fit_normalization(make_dataset(info,[cid,'absent'],'RAW',root,normalization=options))
    assert original['xray']==extended['xray']
    item=make_dataset(info,['absent'],'RAW',root,normalization=options,bounds=extended)[0]
    assert torch.isfinite(item['raw']).all() and torch.count_nonzero(item['chandra'])==0
    with pytest.raises(ValueError,match='No covered training pixels for xray'):
        fit_normalization(make_dataset(info,['absent'],'RAW',root,normalization=options))


def test_dual_encoder_gates_effective_features():
    torch.set_num_threads(1);torch.manual_seed(42)
    model=DualEncoderCNN((3,128,128));model.eval()
    x=torch.randn(2,3,128,128);x[:,2]=0;x[1,2]=1
    captured=[]
    hook=model.classifier.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
    with torch.no_grad():
        original=model(x)
        changed=x.clone();changed[0,1]=1000
        repeated=model(changed)
    hook.remove()
    start=model.radio_encoder.output_dim
    assert torch.count_nonzero(captured[0][0,start:-1])==0
    assert torch.count_nonzero(captured[0][1,start:-1])>0
    assert torch.equal(original[0],repeated[0])


def test_extension_only_adds_train_and_keeps_original_folds():
    base=population(); folds=prepare_folds(base);before=deepcopy(folds)
    expanded=deepcopy(base)
    expanded['extra']=dict(cluster_id='extra',labels=['RR'],rr_rh_known_mask=[1,0])
    out=extend_folds(base,expanded,folds)
    assert folds==before
    for old,new in zip(folds,out):
        assert new['train']==old['train']+['extra']
        assert new['validation']==old['validation'] and new['test']==old['test']
        assert 'extra' not in new['validation']+new['test']
    bad=deepcopy(expanded);bad['extra']['cluster_id']='0'
    with pytest.raises(ValueError,match='Duplicate'):extend_folds(base,bad,folds)
    with pytest.raises(ValueError):extend_folds(base,{},folds)


def test_four_experiments_only_differ_in_cohort_and_radio():
    from dataclasses import asdict
    configs=[asdict(configuration(p,c)) for c in ('paired','expanded') for p in ('RAW','T50kpcSUB')]
    for c in configs:
        assert c['model']=='dual_encoder_cnn' and c['transfer'] is None
        assert c['missing_modality_policy']=='optional'
        c.pop('name');c.pop('radio_source')
    assert all(c==configs[0] for c in configs)


def test_common_optimizer_budget_and_sampling_reproducibility():
    from clusterprep.optional_chandra_cohort import budgeted_loader
    a=budgeted_loader(list(range(39)),42)
    b=budgeted_loader(list(range(128)),42)
    twin=budgeted_loader(list(range(128)),42)
    assert len(a)==len(b)==3
    first=torch.cat(list(b));same=torch.cat(list(twin))
    assert len(first)==48 and torch.equal(first,same)
    assert first.max()<128 and first.min()>=0
    assert not torch.equal(first,torch.cat(list(b)))
