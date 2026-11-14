"""Beam geometry, unit conversions, native RMS, missing inputs and safe resume."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from clusterprep.radio_resolution_diagnostics import (
    canonical_unit, sampling_class, grid_metrics, beam_metrics, native_noise,
    measure_source, pair_metrics, read_config, run_diagnostic, DEFAULTS,
)
from clusterprep.common_wcs_validation import fingerprint


def wcs(scale=2., kind='CDELT', anisotropy=1.):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN','DEC--TAN']
    w.wcs.crpix = [64.5,64.5]; w.wcs.crval = [359.999,70]
    w.wcs.cdelt = [-scale/3600,scale*anisotropy/3600]
    angle = np.deg2rad(23)
    pc = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    if kind == 'PC':
        w.wcs.pc = pc
    if kind == 'CD':
        w.wcs.cd = np.diag(w.wcs.cdelt) @ pc
    return w


def header(major=12., minor=8., angle=0.):
    return dict(BMAJ=major/3600,BMIN=minor/3600,BPA=angle,BUNIT='JY/BEAM')


@pytest.mark.parametrize('unit,factor', [('JY/BEAM',1),('mJy/beam',1e-3),('MJy/beam',1e6),('uJy/beam',1e-6),('mJy beam-1',1e-3)])
def test_units(unit,factor):
    scale, name = canonical_unit(unit)
    assert scale == pytest.approx(factor)
    assert name == 'Jy / beam'


@pytest.mark.parametrize('unit', ['',None,'counts','Jy/pixel nonsense'])
def test_invalid_units(unit):
    with pytest.raises((ValueError, TypeError)):
        canonical_unit(unit)


@pytest.mark.parametrize('value,expected', [(0,'unknown'),(float('nan'),'unknown'),(.999,'severe_lt1'),
                                           (1,'probable_1to2'),(2,'limited_2to3'),(3,'comfortable_ge3')])
def test_thresholds(value,expected):
    assert sampling_class(value)==expected


@pytest.mark.parametrize('kind',['CDELT','PC','CD'])
def test_isotropic_beam_area_and_linear_count(kind):
    target=wcs(kind=kind)
    grid=grid_metrics(target,(128,128)); beam=beam_metrics(header(),target,(128,128),grid)
    assert beam['pixels_per_beam_major']==pytest.approx(6,rel=1e-7)
    assert beam['pixels_per_beam_minor']==pytest.approx(4,rel=1e-7)
    assert beam['beam_area_arcsec2']==pytest.approx(np.pi/(4*np.log(2))*12*8)
    assert grid['fov_area_arcsec2']==pytest.approx((128*2)**2,rel=1e-5)
    assert beam['beams_area']==pytest.approx((256**2)/(np.pi/(4*np.log(2))*12*8),rel=1e-5)
    assert beam['beams_width_over_bmaj']==pytest.approx(256/12,rel=1e-5)
    assert beam['beams_area'] > beam['beams_width_over_bmaj']*10


def test_anisotropy_and_beam_angle():
    target=wcs(scale=2,anisotropy=2)
    grid=grid_metrics(target,(128,128))
    assert grid['pixel_anisotropy']==pytest.approx(2,rel=1e-7)
    # Major north: 12 / 4 = 3 pix ; minor east: 8 / 2 = 4 pix.
    b=beam_metrics(header(angle=0),target,(128,128),grid)
    assert b['pixels_per_beam_minor']==pytest.approx(3,rel=1e-7)
    assert b['pixels_per_beam_major']==pytest.approx(4,rel=1e-7)
    assert b['pixels_per_beam_minor_direction']==pytest.approx(4,rel=1e-7)
    # Rotate beam by 90°: major east =6 pix ; minor north =2 pix.
    b=beam_metrics(header(angle=90),target,(128,128),grid)
    assert b['pixels_per_beam_minor']==pytest.approx(2,rel=1e-7)
    assert b['pixels_per_beam_major']==pytest.approx(6,rel=1e-7)


@pytest.mark.parametrize('beam',[{},header(minor=0),header(major=2,minor=8),header(angle=np.nan)])
def test_invalid_beam(beam):
    with pytest.raises(ValueError):
        beam_metrics(beam,wcs(),(128,128))


def test_native_robust_noise():
    rng=np.random.default_rng(42)
    image=rng.normal(5,2,(256,256))
    image[64:192,64:192]+=1000  # central emission excluded
    image[10:20,10:20]+=1000   # bright peripheral source clipped
    image[:4,:]=np.nan
    result=native_noise(image,wcs())
    assert result['rms_native']==pytest.approx(2,rel=.03)
    assert result['background_median']==pytest.approx(5,abs=.04)
    assert result['background_retained'] < result['background_candidates']
    with pytest.raises(ValueError):
        native_noise(np.zeros((128,128)),wcs())
    with pytest.raises(ValueError):
        native_noise(np.full((128,128),np.nan),wcs())


def test_incompatible_units_never_produce_ratio():
    base=dict(cluster_id='test',source_exists=True,beam_valid=True,noise_valid=True,unit_valid=True,
              pixels_per_beam_minor=1.5,beam_minor_arcsec=8,beam_area_arcsec2=100,
              sampling_class='probable_1to2',coverage=.9,rms_canonical=.002)
    raw={**base,'modality':'RAW','canonical_unit':'Jy / beam'}
    sub={**base,'modality':'T50kpcSUB','canonical_unit':'Jy / sr','pixels_per_beam_minor':3,'coverage':.6}
    result=pair_metrics(pd.DataFrame([raw,sub])).iloc[0]
    assert not result.units_compatible and not result.usable_pair
    assert result.rms_ratio is None
    assert result.raw_under2_tapered_ge2 and result.large_coverage_difference
    sub['canonical_unit']='Jy / beam'; sub['rms_canonical']=.006
    result=pair_metrics(pd.DataFrame([raw,sub])).iloc[0]
    assert result.rms_ratio==pytest.approx(3)


def test_missing_and_invalid_metadata(tmp_path):
    target=wcs(); grid=grid_metrics(target,(128,128))
    missing=measure_source(tmp_path/'absent.fits',target,(128,128),grid)
    assert not missing['source_exists'] and not missing['beam_valid']
    path=tmp_path/'image.fits'
    fits.PrimaryHDU(np.random.default_rng(2).normal(size=(128,128)),target.to_header()).writeto(path)
    row=measure_source(path,target,(128,128),grid)
    assert row['noise_valid'] and not row['unit_valid'] and not row['beam_valid']
    assert row['beam_error'] and row['unit_error']


def test_resume_and_no_input_writes(tmp_path,monkeypatch):
    import clusterprep.radio_resolution_diagnostics as module
    monkeypatch.setattr(module,'plot_diagnostics',lambda *args:None)
    monkeypatch.setattr(module,'plot_gallery',lambda *args:[])
    validation=tmp_path/'validation'; directory=validation/'processed/test'; directory.mkdir(parents=True)
    source=tmp_path/'source.fits'; target=wcs()
    h=target.to_header(); h.update(header())
    fits.PrimaryHDU(np.random.default_rng(9).normal(size=(128,128)),h).writeto(source)
    meta=dict(wcs=dict(target.to_header()), sources={'RAW':str(source),'T50kpcSUB':str(tmp_path/'missing.fits')},
              identity={'settings':{'config':{'output_size':128}}})
    (directory/'metadata.json').write_text(json.dumps(meta))
    (validation/'configuration.json').write_text('{}')
    pd.DataFrame([dict(cluster_id='test',redshift=.2,fov_method='r500',fov_arcmin=256/60,
                       pixel_scale_arcsec=2,coverage_raw=1.,coverage_t50kpcsub=np.nan)]).to_csv(validation/'summary.csv',index=False)
    inputs=[source,*validation.rglob('*json'),validation/'summary.csv']
    before={str(p):fingerprint(p) for p in inputs}
    config={**DEFAULTS,'validation_dir':str(validation),'output_dir':str(tmp_path/'diagnostic')}
    _,_,first=run_diagnostic(config,progress=False); assert first['reused']==0
    metrics,pairs,second=run_diagnostic(config,progress=False); assert second['reused']==2
    assert len(metrics)==2 and not pairs.iloc[0].both_sources
    assert before=={str(p):fingerprint(p) for p in inputs}
    # Content change invalidates only that source measurement.
    with fits.open(source,mode='update') as hd: hd[0].data *= 2
    _,_,third=run_diagnostic(config,progress=False); assert third['reused']==1
    with pytest.raises(ValueError,match='independent'):
        run_diagnostic({**config,'output_dir':str(validation/'bad')},progress=False)
    # A broken target grid must not masquerade as a missing native FITS.
    meta.pop('wcs')
    (directory/'metadata.json').write_text(json.dumps(meta))
    broken,_,_=run_diagnostic(config,progress=False)
    raw=broken[broken.modality=='RAW'].iloc[0]
    assert raw.source_exists and raw.noise_valid and raw.unit_valid
    assert raw.grid_error and not raw.beam_valid


def test_config_validation(tmp_path):
    p=tmp_path/'config.json';p.write_text(json.dumps({'coverage_threshold':2}))
    with pytest.raises(ValueError): read_config(p,tmp_path)
    p.write_text(json.dumps({'coverage_difference_threshold':.15}))
    assert read_config(p,tmp_path)['coverage_difference_threshold']==.15
