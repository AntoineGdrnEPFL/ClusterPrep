import numpy as np
import pytest
from astropy.wcs import WCS
from scipy.signal import fftconvolve

from clusterprep.radio_noise import add_radio_noise, beam_covariance, local_jacobian, _kernel
from clusterprep.radio_noise_diagnostics import (lag_one_correlations, autocorrelation_window,
    kernel_autocorrelation, beam_diagnostics, background_diagnostics, pipeline_residuals)


def test_lags_mask_and_affine_scaling():
    x=np.random.default_rng(18).normal(size=(128,128))
    x[:5,:9]=np.nan
    a=lag_one_correlations(x)
    b=lag_one_correlations(5*x+11)
    assert a['pairs_x']==128*127-5*9
    assert a['pairs_y']==127*128-5*9
    assert a['x']==pytest.approx(b['x'])
    assert a['y']==pytest.approx(b['y'])
    ac=autocorrelation_window(x,20)
    assert ac.shape==(41,41)
    assert ac[20,20]==pytest.approx(1)
    np.testing.assert_allclose(ac,ac[::-1,::-1],atol=1e-12)
    assert np.isnan(autocorrelation_window(np.zeros((64,64)))).all()
    assert np.isnan(lag_one_correlations(np.zeros((64,64)))['x'])


@pytest.mark.parametrize('covariance', [[[9,2],[2,3]], [[.01,0],[0,.01]], [[4,0],[0,4]]])
def test_measured_vs_actual_kernel_and_convolution_order(covariance):
    context=dict(rms=2,covariance=covariance)
    cfg=dict(enabled=True,method='beam_correlated',probability=1,amplitude_range=[.3,.3])
    seed=13
    rng=np.random.default_rng(seed)
    rng.random();rng.uniform(.3,.3)
    k=_kernel(tuple(np.array(covariance).ravel()))
    shape=(512,512)
    white=rng.standard_normal(tuple(n+m-1 for n,m in zip(shape,k.shape)))
    convolution=fftconvolve(white,k,mode='valid')
    expected=(convolution-convolution.mean())/convolution.std()*.6
    actual=add_radio_noise(np.zeros(shape,dtype='float32'),context,cfg,np.random.default_rng(seed))
    np.testing.assert_allclose(actual,expected,atol=2e-7)
    rho=lag_one_correlations(actual)
    before=lag_one_correlations(convolution)
    assert rho['x']==pytest.approx(before['x'],abs=1e-8)
    assert rho['y']==pytest.approx(before['y'],abs=1e-8)
    ac=kernel_autocorrelation(context,20)
    assert rho['x']==pytest.approx(ac[20,21],abs=.025)
    assert rho['y']==pytest.approx(ac[21,20],abs=.025)


@pytest.mark.parametrize('matrix', [[[-.001,0],[0,.002]], [[-.001,.0003],[.0002,.002]], [[.001,0],[0,-.001]]])
def test_wcs_independent_endpoints(matrix):
    wcs=WCS(naxis=2)
    wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    wcs.wcs.crpix=[64.5,64.5]
    wcs.wcs.crval=[359.999,70]
    wcs.wcs.cd=np.array(matrix)
    header=dict(BMAJ=.008,BMIN=.004,BPA=37)
    context=dict(metadata=header,covariance=beam_covariance(header,wcs,(128,128)),
                 jacobian=local_jacobian(wcs,(128,128)))
    check=beam_diagnostics(context,wcs,(128,128))
    assert check['endpoint_relative_error']<1e-5
    assert check['fwhm_major_px']>=check['fwhm_minor_px']>0


def test_gaussian_sources_background_sensitivity():
    rng=np.random.default_rng(6)
    clean=rng.normal(0,1,(256,256))
    y,x=np.indices(clean.shape)
    contaminated=clean.copy()
    source=np.zeros_like(clean)
    for cx,cy in [(20,20),(40,200),(230,90),(180,230)]:
        source+=80*np.exp(-((x-cx)**2+(y-cy)**2)/(2*2**2))
    contaminated+=source
    d=background_diagnostics(contaminated)
    assert d['rms_after']==pytest.approx(1,rel=.04)
    assert d['rms_buffer2']==pytest.approx(1,rel=.04)
    cores=d['candidate'] & (source>5)
    assert (d['rejected'] & cores).sum()/cores.sum()>.98
    assert not np.any(d['kept'] & d['rejected'])
    assert d['n_positive_rejected']>0


def test_residual_diagnostics_match_dataset(tmp_path):
    from astropy.io import fits
    from clusterprep.dataset import ClusterDataset
    from clusterprep.preprocessing import preprocess
    from clusterprep.radio_noise import prepare_noise_context
    wcs=WCS(naxis=2)
    wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    wcs.wcs.cdelt=[-.001,.001]
    header=wcs.to_header();header.update(BMAJ=.006,BMIN=.003,BPA=25)
    image=np.random.default_rng(22).normal(size=(64,64)).astype('float32')
    image[:4,:4]=np.nan
    path=tmp_path/'raw.fits';fits.PrimaryHDU(image,header).writeto(path)
    info=dict(cluster_id='a',raw_path=str(path))
    cfg=dict(enabled=True,method='beam_correlated',probability=1,amplitude_range=[.5,.5])
    images=preprocess(info,'raw',64)
    context=prepare_noise_context(images,info)
    np.random.seed(21)
    augmented=add_radio_noise(images['raw'],context,cfg)
    for variant,angle,flip in [(0,0,False),(2,30,False),(3,30,True),(6,90,False)]:
        ds=ClusterDataset({'a':info},['a'],[],size=64,augment=True,processed_dir=None,
            include_targets=False,training=True,radio_noise=cfg)
        clean=ClusterDataset({'a':info},['a'],[],size=64,augment=True,processed_dir=None,
            include_targets=False)
        np.random.seed(21)
        actual=(ds[variant]['raw']-clean[variant]['raw']).numpy()[0]
        stages=pipeline_residuals(images['raw'],augmented,angle=angle,flip=flip)
        valid=np.isfinite(stages['geometry'])
        np.testing.assert_array_equal(actual[valid],stages['geometry'][valid])
