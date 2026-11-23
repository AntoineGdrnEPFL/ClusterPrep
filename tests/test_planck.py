"""Planck contracts with small, deterministic real HEALPix FITS fixtures."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
import astropy.units as u
from clusterprep.planck import project, _read_map, source_identity
from clusterprep.dataset import ClusterDataset
from clusterprep.modalities import fit_normalization
from clusterprep.training import ExperimentConfig, run_experiment
from clusterprep.models import build_model
from test_matched_dataset import catalog


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def sky(tmp_path):
    nside = 16
    hp = HEALPix(nside=nside, order='ring', frame='galactic')
    coords = hp.healpix_to_skycoord(np.arange(12*nside*nside))
    values = (2 + np.sin(coords.l.rad) * np.cos(coords.b.rad)) * 1e-5
    table = fits.BinTableHDU.from_columns([fits.Column(name='FULL', format='D', unit='y', array=values)])
    table.header.update(PIXTYPE='HEALPIX', NSIDE=nside, ORDERING='RING', COORDSYS='GALACTIC', BEAMSIZE=10)
    source = tmp_path/'sky.fits'
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(source)
    return dict(source_path=str(source), cache_dir=str(tmp_path/'planck'))


@pytest.fixture
def sources(tmp_path):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [180, 45]
    w.wcs.crpix = [16.5, 16.5]
    w.wcs.cdelt = [-.1, .1]
    result = {}
    for i in range(12):
        path = tmp_path/f'radio{i}.fits'
        fits.writeto(path, np.arange(1024, dtype='float32').reshape(32,32)+i, w.to_header())
        result[str(i)] = dict(cluster_id=f'c{i}', raw_path=str(path),
            chandra_path=str(path) if i%3 else None,
            labels={'NDE'} if i%2 else {'DE','RR','RH'})
    return result


def ds(sources, sky, keys=None, **kw):
    return ClusterDataset(sources, list(sources) if keys is None else keys, ['DE','NDE'],
        size=8, processed_dir=None, modalities=kw.pop('modalities', ['radio','planck']),
        planck=sky, **kw)


def test_projection_cache_coordinates(sky, sources):
    w = WCS(fits.getheader(sources['0']['raw_path']))
    a = project(sky,w,(32,32))
    assert a['cache_status']=='created'
    b = project(sky,w,(32,32))
    assert b['cache_status']=='reused'
    np.testing.assert_array_equal(a['planck'],b['planck'])
    # Independent analytic Galactic field evaluated at equatorial target positions.
    y,x=np.indices((32,32))
    gal = w.pixel_to_world(x,y).galactic
    truth = (2+np.sin(gal.l.rad)*np.cos(gal.b.rad))*1e-5
    np.testing.assert_allclose(a['planck'],truth,atol=3e-8)
    assert WCS(fits.getheader(a['cache_path'])).celestial.wcs.compare(w.wcs)
    assert fits.getheader(a['cache_path'])['BUNIT']=='y'
    Path(a['cache_path']).write_bytes(b'corrupt')
    assert project(sky,w,(32,32))['cache_status']=='rebuilt'
    assert project({**sky,'force_rebuild':True},w,(32,32))['cache_status']=='rebuilt'
    assert project(sky,w,(16,16))['cache_path']!=a['cache_path']
    w.wcs.crval[0]+=1
    assert project(sky,w,(32,32))['cache_path']!=a['cache_path']
    assert _read_map.cache_info().hits>0


def test_invalid_and_source_invalidation(sky, sources):
    w = WCS(fits.getheader(sources['0']['raw_path']))
    a=project(sky,w,(32,32))
    with fits.open(sky['source_path'], mode='update') as hdus:
        hdus[1].data['FULL'][:]=-1.6375e30
    b=project(sky,w,(32,32))
    assert b['cache_path']!=a['cache_path']
    assert not b['planck_mask'].any() and np.isnan(b['planck']).all()
    sample=ds(sources,sky,['0'],normalization={'planck':{'method':'none'}})[0]
    assert not sample['availability'][-1] and not sample['planck'].any()


def test_geometry_and_statistics(sky,sources):
    train=ds(sources,sky,['0','1'],mode='multimodal')
    bounds=fit_normalization(train)
    with fits.open(sources['11']['raw_path'],mode='update') as hdus:
        hdus[0].header['CRVAL1']=12
    assert fit_normalization(train)==bounds
    with pytest.raises(ValueError,match='bounds missing'):
        train[0]
    train.normalization_bounds=bounds
    base=train[0]
    augmented=ds(sources,sky,['0'],training=True,augment=True,normalization_bounds=bounds)
    from clusterprep.augmentation import transform_aligned
    for index in range(24):
        sample=augmented[index]
        angle,flip=30*(index//2),bool(index%2)
        for name in ['raw','planck','planck_mask']:
            torch.testing.assert_close(sample[name],transform_aligned(base[name],angle,flip))
    # No per-augmentation copies: only two source reference grids have been cached.
    assert len(list(Path(sky['cache_dir']).glob('*.fits')))==2


def test_modes(sky,sources):
    assert len(ds(sources,sky,mode='pair'))==12
    assert len(ds(sources,sky,mode='pair',modalities=['radio','xray','planck']))==8
    multi=ds(sources,sky,mode='multimodal',modalities=['radio','xray','planck'],
             normalization={'planck':{'method':'none'}})
    assert len(multi)==12
    assert multi[0]['availability'].tolist()==[True,False,True]
    assert not multi[0]['chandra_mask'].any()


@pytest.mark.parametrize('source',['RAW','T50kpcSUB'])
def test_matched(sky,catalog,source):
    info,cid,root=catalog
    dataset=ClusterDataset(info,[cid],['RR','RH'],task='rr_rh',size=128,
        mode='multimodal',modalities=['radio','xray','planck'],planck=sky,
        normalization={'planck':{'method':'none'}},processed_dir=root,
        radio_preprocessing='matched_resolution',radio_source=source)
    images=dataset._modality_images(dataset.data_info[cid])
    sample=dataset[0]
    assert sample['image'].shape==(5,128,128)
    assert torch.isfinite(sample['image']).all()
    assert images['planck'].shape==images['raw'].shape
    expected=project(sky,images['wcs'],(128,128),'independent')
    np.testing.assert_allclose(images['planck'],expected['planck'])


@pytest.mark.parametrize('model,modalities',[('dual_encoder_cnn',['radio','planck']),
                                           ('multi_encoder_cnn',['radio','xray','planck'])])
def test_training_reload(sky,sources,tmp_path,model,modalities):
    config=ExperimentConfig(name=model,model=model,mode='multimodal',task='rr_rh',
        modalities=modalities,planck=sky,size=8,device='cpu',batch_size=2,epochs=1,
        augment=False,processed_dir=None,verbose=False)
    splits=([str(i) for i in range(4)],[str(i) for i in range(4,8)],[str(i) for i in range(8,12)])
    run=run_experiment(sources,config,tmp_path/'runs',splits=splits)
    assert json.loads((run/'status.json').read_text())['status']=='completed'
    metrics=json.loads((run/'metrics.json').read_text())
    assert metrics['test'] is not None and metrics['split_counts']['test']==4
    assert (run/'test_predictions.csv').exists()
    saved=ExperimentConfig(**json.loads((run/'config.json').read_text())['config'])
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    restored=build_model(saved,2).eval()
    restored.load_state_dict(checkpoint['model_state_dict'])
    sample=ds(sources,sky,['8'],modalities=modalities,normalization_bounds=saved.normalization_bounds)[0]
    assert torch.isfinite(restored(sample['image'][None])).all()
    assert json.loads((run/'normalization.json').read_text())['train_keys']==splits[0]


def test_transfer_preserves_new_encoder(sky,tmp_path):
    from dataclasses import asdict
    from clusterprep.modular_transfer import initialize_transfer,ModularTransferConfig,resolve_module
    old=ExperimentConfig(name='old',model='dual_encoder_cnn',mode='multimodal',size=8)
    source=build_model(old,2)
    path=tmp_path/'old.pt'
    torch.save(dict(config=asdict(old),class_names=['DE','NDE'],model_state_dict=source.state_dict()),path)
    cfg=replace(old,name='new',model='multi_encoder_cnn',modalities=['radio','xray','planck'],planck=sky)
    target=build_model(cfg,2)
    before={k:v.clone() for k,v in target.encoders['planck'].state_dict().items()}
    initialize_transfer(target,ModularTransferConfig(enabled=True,sources={'old':{'checkpoint':str(path)}},
        mappings=[{'source':'old','from':m,'to':m} for m in ['radio','xray']]))
    for name in ['radio','xray']:
        for k,v in resolve_module(source,name).state_dict().items():
            torch.testing.assert_close(v,resolve_module(target,name).state_dict()[k])
    for k,v in before.items():
        torch.testing.assert_close(v,target.encoders['planck'].state_dict()[k])


def test_none_and_asinh(sky,sources):
    for method in ['none','percentile_asinh']:
        dataset=ds(sources,sky,['0'],normalization={'planck':{'method':method}})
        dataset.normalization_bounds=fit_normalization(dataset)
        assert torch.isfinite(dataset[0]['planck']).all()
    with pytest.raises(ValueError,match='train statistics'):
        ds(sources,sky,normalization={'planck':{'scope':'image'}})


def test_missing_features_neutral(sky):
    for name,mods in [('dual_encoder_cnn',['radio','planck']),('multi_encoder_cnn',['radio','xray','planck'])]:
        config=ExperimentConfig(name='neutral',model=name,size=8,modalities=mods,planck=sky)
        model=build_model(config,2).eval()
        x=torch.randn(2,3 if len(mods)==2 else 5,8,8)
        x[:,2::2]=0
        a=model(x)
        x[:,1::2]=123
        torch.testing.assert_close(a,model(x))


def test_invalid_metadata_and_empty_cache(sky,sources):
    w=WCS(fits.getheader(sources['0']['raw_path']))
    result=project(sky,w,(32,32))
    fits.PrimaryHDU().writeto(result['cache_path'],overwrite=True)
    assert project(sky,w,(32,32))['cache_status']=='rebuilt'
    with fits.open(sky['source_path'],mode='update') as hdus:
        hdus[1].header['NSIDE']=8
    with pytest.raises(RuntimeError,match='NSIDE'):
        project(sky,w,(32,32))


def test_noncanonical_wcs_cache(sky):
    # Sliced radio WCS can carry ancillary attributes not serialized by to_header.
    w=WCS(naxis=4)
    w.wcs.ctype=['RA---SIN','DEC--SIN','FREQ','STOKES']
    w.wcs.crval=[-163.,26.,1e8,1.]
    w.wcs.cdelt=[-.01,.01,1.,1.]
    celestial=w.celestial
    first=project(sky,celestial,(8,8))
    assert project(sky,celestial,(8,8))['cache_status']=='reused'
    with fits.open(first['cache_path'],mode='update') as hdus:
        hdus[0].header['CRVAL1']=42.
    assert project(sky,celestial,(8,8))['cache_status']=='rebuilt'


def test_zoom_masks(sky,sources):
    from clusterprep.augmentation import transform_aligned,decode_variant
    zoom=dict(enabled=True,factors=[1.,1.5,.8])
    plain=ds(sources,sky,['0'],normalization={'planck':{'method':'none'}})[0]
    augmented=ds(sources,sky,['0'],normalization={'planck':{'method':'none'}},
                 augment=True,training=True,zoom=zoom)
    for i in [24,35,48,71]:
        angle,flip,scale=decode_variant(i,zoom)
        expected=transform_aligned(torch.cat([plain['planck'],plain['planck'],plain['planck_mask']]),
                                   angle,flip,scale,zoom)
        torch.testing.assert_close(augmented[i]['planck'],expected[1:2])
        torch.testing.assert_close(augmented[i]['planck_mask'],expected[2:3])
        assert set(augmented[i]['planck_mask'].unique().tolist()) <= {0.,1.}


def test_pair_rejects_uncovered_planck(sky,sources):
    with fits.open(sky['source_path'],mode='update') as hdus:
        hdus[1].data['FULL'][:]=np.nan
    dataset=ds(sources,sky,['0'],mode='pair',normalization={'planck':{'method':'none'}})
    with pytest.raises(ValueError,match='No valid Planck coverage'):
        dataset[0]


@pytest.mark.parametrize('geometry',['historical','matched_resolution'])
@pytest.mark.parametrize('experiment',['A','C'])
def test_cli_final_test(geometry,experiment,monkeypatch,capsys):
    from clusterprep import matched_runner as runner
    calls=[]
    monkeypatch.setattr(runner,'run_experiment',lambda *a,**kw:calls.append((a,kw)))
    root=Path(__file__).resolve().parents[1]
    for fold in range(5):
        path=root/f'configs/planck/{geometry}/{experiment}/fold_{fold}.json'
        spec=json.loads(path.read_text())
        runner.run_config_suite(spec,path,dry_run=True)
        result=json.loads(capsys.readouterr().out)
        assert result['evaluate_test'] and result['test']==13 and result['reserved_test']==0
        assert not calls
    runner.run_config_suite(spec,path,quiet=True)
    assert calls[0][0][1].evaluate_test
    assert len(calls[0][1]['splits'][2])==13
