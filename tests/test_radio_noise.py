import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.wcs import WCS
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from clusterprep.dataset import ClusterDataset
from clusterprep.preprocessing import preprocess, normalize
from clusterprep.processed import load_processed
from clusterprep.radio_noise import (validate_radio_noise, estimate_rms, beam_covariance,
    local_jacobian, _kernel, prepare_noise_context, add_radio_noise)
from clusterprep.training import ExperimentConfig, _seed_worker


def config(method='gaussian_pixel', **kw):
    return dict(enabled=True, method=method, probability=1., amplitude_range=[.5, .5], **kw)


@pytest.fixture
def source(tmp_path):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crpix = [64.5, 64.5]
    wcs.wcs.crval = [180, 45]
    wcs.wcs.cdelt = [-.001, .001]
    header = wcs.to_header()
    header.update(BMAJ=.008, BMIN=.004, BPA=30., BUNIT='Jy/beam')
    data = np.random.default_rng(5).normal(0, .02, (128, 128)).astype('float32')
    data[:8, :8] = np.nan
    path = tmp_path/'radio.fits'
    fits.PrimaryHDU(data, header).writeto(path)
    return dict(cluster_id='a', raw_path=str(path), chandra_path=str(path), labels={'DE'})


@pytest.mark.parametrize('bad', [dict(method='both'), dict(probability=2), dict(probability=float('nan')),
    dict(amplitude_range=[-1, 1]), dict(amplitude_range=[2, 1]), dict(enabled='false'),
    dict(rms_source='catalog'), dict(missing_beam_policy='silent'), dict(rms_estimation={})])
def test_config(bad):
    with pytest.raises(ValueError):
        ExperimentConfig(name='test', augmentation={'radio_noise': bad})


def test_scattering_rejected():
    with pytest.raises(ValueError, match='scattering'):
        ExperimentConfig(name='test', model='dual_ssn', augmentation={'radio_noise': config()})


def test_rms_robust_units():
    image = np.random.default_rng(123).normal(0, 2, (256, 256))
    image[20:40, 20:40] += 200
    image[80:140, 80:140] += 300
    assert estimate_rms(image) == pytest.approx(2, rel=.03)
    assert estimate_rms(image*1000) == pytest.approx(estimate_rms(image)*1000)
    with pytest.raises(ValueError):
        estimate_rms(np.zeros((128, 128)))
    with pytest.raises(ValueError):
        estimate_rms(np.full((128, 128), np.nan))


def test_wcs_orientation_and_reduction(source):
    images = preprocess(source, 'raw', 64)
    cov = beam_covariance(images['radio_metadata'], images['wcs'], (64, 64))
    eig = np.linalg.eigvalsh(cov)
    np.testing.assert_allclose(np.sqrt(eig)*np.sqrt(8*np.log(2)), [2,4], rtol=.001)
    assert cov[0,1] < 0  # positive BPA, negative RA pixel direction
    wcs = images['wcs'].deepcopy()
    theta = np.deg2rad(40)
    wcs.wcs.pc = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    transformed = beam_covariance(images['radio_metadata'], wcs, (64,64))
    jac = local_jacobian(wcs, (64,64))
    original_jac = local_jacobian(images['wcs'], (64,64))
    np.testing.assert_allclose(jac@transformed@jac.T, original_jac@cov@original_jac.T, rtol=1e-4)
    circular = beam_covariance(dict(BMAJ=.004,BMIN=.004,BPA=80), images['wcs'], (64,64))
    np.testing.assert_allclose(circular, np.eye(2)*circular[0,0], atol=1e-7)


@pytest.mark.parametrize('method', ['gaussian_pixel', 'beam_correlated'])
def test_statistics_and_rng(method):
    image = np.zeros((256,256), dtype='float32')
    image[:4,:4] = np.nan
    context = dict(rms=2., covariance=[[9.,0.],[0.,1.]])
    a = add_radio_noise(image, context, config(method), np.random.default_rng(4))
    b = add_radio_noise(image, context, config(method), np.random.default_rng(4))
    np.testing.assert_equal(a,b)
    assert not np.allclose(a, add_radio_noise(image, context, config(method), np.random.default_rng(5)), equal_nan=True)
    assert np.isnan(a[:4,:4]).all()
    assert a.dtype == np.float32
    assert abs(np.nanmean(a)) < .02
    assert np.nanstd(a) == pytest.approx(1, rel=.02)
    corr_x = np.corrcoef(a[8:,:-1].ravel(), a[8:,1:].ravel())[0,1]
    corr_y = np.corrcoef(a[8:-1,:].ravel(), a[9:,:].ravel())[0,1]
    if method == 'gaussian_pixel':
        assert abs(corr_x) < .02 and abs(corr_y) < .02
    else:
        assert corr_x > corr_y + .1
    assert np.nanmax(image) == 0


@pytest.mark.parametrize('policy', ['raise', 'skip', 'gaussian_pixel'])
def test_missing_beam(policy):
    image = np.zeros((64,64), dtype='float32')
    args = (image, dict(rms=1), config('beam_correlated', missing_beam_policy=policy))
    if policy == 'raise':
        with pytest.raises(ValueError): add_radio_noise(*args)
    else:
        with pytest.warns(RuntimeWarning, match=policy): out = add_radio_noise(*args)
        assert bool(out.any()) == (policy == 'gaussian_pixel')


@pytest.mark.parametrize('mode', ['raw','pair','multimodal'])
def test_dataset_order_modalities_cache_and_eval(source, tmp_path, mode):
    args = ({'a':source}, ['a'], ['DE','NDE'])
    kw = dict(mode=mode,size=64,processed_dir=tmp_path/'cache',cache=True,augment=True)
    clean = ClusterDataset(*args, **kw)
    disabled = ClusterDataset(*args, **kw,training=True,radio_noise={'enabled':False})
    evaluation = ClusterDataset(*args, **kw,training=False,radio_noise=config())
    noisy = ClusterDataset(*args, **kw,training=True,radio_noise=config())
    for other in (disabled,evaluation):
        torch.testing.assert_close(clean[3]['raw'], other[3]['raw'], rtol=0,atol=0)
    images = load_processed(source, 'raw' if mode=='raw' else 'pair',64,tmp_path/'cache')
    context = prepare_noise_context(images,source,tmp_path/'cache')
    np.random.seed(12)
    expected = add_radio_noise(images['raw'],context,config())
    bounds = np.quantile(images['raw'][np.isfinite(images['raw'])].astype(float),[.3,.99])
    expected = torch.from_numpy(normalize(expected,bounds=bounds))[None]
    np.random.seed(12)
    sample = noisy[0]
    torch.testing.assert_close(sample['raw'],expected,atol=0,rtol=0)
    if mode != 'raw':
        for name in ('chandra','chandra_mask','label'):
            torch.testing.assert_close(sample[name],clean[0][name],rtol=0,atol=0)
    with patch('clusterprep.dataset.load_processed',side_effect=AssertionError('reopen')):
        second = noisy[0]
        assert not torch.equal(sample['raw'],second['raw'])
    np.random.seed(12)
    torch.testing.assert_close(noisy[1]['raw'],TF.hflip(expected))
    with patch('clusterprep.radio_noise.estimate_rms', side_effect=AssertionError('reestimate')):
        assert prepare_noise_context(images,source,tmp_path/'cache') == context


def test_kernel_and_metadata_cache(source,tmp_path):
    first=load_processed(source,'raw',64,tmp_path/'cache')
    second=load_processed(source,'raw',64,tmp_path/'cache')
    assert first['radio_metadata']==second['radio_metadata']
    context=prepare_noise_context(second)
    _kernel.cache_clear()
    for i in range(3): add_radio_noise(first['raw'],context,config('beam_correlated'))
    assert _kernel.cache_info().misses == 1
    assert _kernel.cache_info().hits == 2


def _numpy_collate(batch):
    # Exercise worker seeding without depending on macOS torch_shm_manager.
    return np.stack([b['raw'].numpy() for b in batch])


@pytest.mark.parametrize('workers',[0,2])
def test_worker_reproducibility(source,workers):
    def sequence(seed):
        np.random.seed(seed)
        ds=ClusterDataset({'a':source},['a'],['DE','NDE'],size=64,processed_dir=None,
                          training=True,radio_noise=config(),augment=True)
        loader=DataLoader(ds,batch_size=3,num_workers=workers,worker_init_fn=_seed_worker,
                          generator=torch.Generator().manual_seed(seed), collate_fn=_numpy_collate)
        return torch.from_numpy(np.concatenate(list(loader)))
    a=sequence(7)
    torch.testing.assert_close(a,sequence(7),atol=0,rtol=0)
    assert not torch.equal(a,sequence(8))


def test_experiment_jsons_are_matched():
    from pathlib import Path
    root=Path(__file__).parents[1]/'configs'
    expected={(model,task) for model in ('fusion','dual_encoder_cnn')
              for task in ('rr_binary','rh_binary')}
    for fold in range(5):
        combined=json.loads((root/'binary_subtypes_multimodal_5folds'/f'fold_{fold}.json').read_text())
        variants=[]
        reference=None
        for method in ['baseline','gaussian_pixel','beam_correlated']:
            spec=json.loads((root/'radio_noise'/method/f'fold_{fold}.json').read_text())
            assert len(spec['experiments'])==4
            assert {(e['model'],e['task']) for e in spec['experiments']}==expected
            assert spec['split']==combined['split']
            assert spec['split']['fold_index']==fold
            assert {k:v for k,v in spec.items() if k not in {'output_dir','experiments'}} == {
                k:v for k,v in combined.items() if k not in {'output_dir','experiments'}}
            clean=[]
            for exp in spec['experiments']:
                ExperimentConfig(**exp)
                noise=exp['augmentation']['radio_noise']
                assert noise['enabled']==(method!='baseline')
                assert noise['method']==('gaussian_pixel' if method=='baseline' else method)
                assert noise['probability']==.5 and noise['amplitude_range']==[0.,.5]
                clean.append({k:v for k,v in exp.items() if k not in {'name','augmentation'}})
            if reference is None:
                reference=clean
            assert clean==reference
            if method!='baseline':
                variants.extend(spec['experiments'])
        assert combined['experiments']==variants


@pytest.mark.parametrize('method',['gaussian_pixel','beam_correlated'])
def test_training_integration_train_only(source,tmp_path,method):
    from clusterprep.training import run_experiment
    info={}
    for i in range(6):
        # Distinct source files and cluster IDs, no shared image across splits.
        import shutil
        path=tmp_path/f'{i}.fits'
        shutil.copyfile(source['raw_path'],path)
        info[str(i)]={**source, 'cluster_id':str(i),'raw_path':str(path),
                      'chandra_path':None,'labels':{'DE' if i%2==0 else 'NDE'}}
    calls=[]
    def dataset(*args,**kwargs):
        ds=ClusterDataset(*args,**kwargs)
        calls.append(ds)
        return ds
    cfg=ExperimentConfig(name=method,size=16,width=2,device='cpu',epochs=1,
        batch_size=2,augment=False,processed_dir=None,verbose=False,
        augmentation={'radio_noise':config(method)})
    threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with patch('clusterprep.training.ClusterDataset',side_effect=dataset):
            run=run_experiment(info,cfg,tmp_path/'runs',splits=(['0','1'],['2','3'],['4','5']))
    finally:
        torch.set_num_threads(threads)
    assert [ds.noise_enabled for ds in calls]==[True,False,False]
    assert json.loads((run/'metrics.json').read_text())['epochs_ran']==1


@pytest.mark.parametrize('metadata', [None, {}, {'BMAJ': .008}])
def test_legacy_cache_metadata_read_once(source,tmp_path,metadata):
    from clusterprep.io import load_celestial_fits
    images=preprocess(source,'raw',64)
    images['radio_metadata'] = metadata  # legacy or incomplete instrumental headers
    with patch('clusterprep.io.load_celestial_fits',wraps=load_celestial_fits) as reader:
        first=prepare_noise_context(images,source,tmp_path/'legacy')
        second=prepare_noise_context(images,source,tmp_path/'legacy')
        assert reader.call_count==1
        assert first==second
        assert set(first['metadata']) == {'BMAJ', 'BMIN', 'BPA', 'BUNIT'}
        assert 'covariance' in first


def test_old_noise_context_rebuilt(source,tmp_path):
    images=preprocess(source,'raw',64)
    context=prepare_noise_context(images,source,tmp_path)
    path=next((tmp_path/'radio_noise').glob('*.json'))
    old=json.loads(path.read_text())
    old['version']=1
    old['context']={'metadata':{}, 'beam_error':'BMAJ/BMIN/BPA absents ou invalides'}
    path.write_text(json.dumps(old))
    assert prepare_noise_context(images,source,tmp_path)==context
    assert json.loads(path.read_text())['version']==2


def test_truly_missing_source_beam_still_raises(source):
    with fits.open(source['raw_path'], mode='update') as hdul:
        for key in ['BMAJ','BMIN','BPA']:
            del hdul[0].header[key]
    images=preprocess(source,'raw',64)
    context=prepare_noise_context(images,source)
    assert 'covariance' not in context
    with pytest.raises(ValueError,match='BMAJ/BMIN/BPA'):
        add_radio_noise(images['raw'],context,config('beam_correlated'))


def test_circular_noise_isotropic():
    out=add_radio_noise(np.zeros((512,512),dtype='float32'),
        dict(rms=1,covariance=[[4,0],[0,4]]),config('beam_correlated'),np.random.default_rng(33))
    cx=np.corrcoef(out[:,:-1].ravel(),out[:,1:].ravel())[0,1]
    cy=np.corrcoef(out[:-1,:].ravel(),out[1:,:].ravel())[0,1]
    assert abs(cx-cy)<.02
    assert np.std(out)==pytest.approx(.5,rel=1e-6)
