"""Cache optimization contracts: identical physical data, bounded deterministic reuse."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.wcs import WCS
from clusterprep.dataset import ClusterDataset
from clusterprep.io import load_celestial_fits, load_celestial_geometry
from clusterprep.planck import aligned_images, project
from clusterprep.preprocessing import preprocess, block_average, centered_raw_window
from test_planck import sky, sources, ds, catalog


def test_reduced_cache_exact_and_reused_without_radio_pixels(sky, sources):
    info=sources['0']; images=preprocess(info,'raw',8)
    raw,_,wcs=load_celestial_fits(info['raw_path'])
    full=project(sky,wcs,raw.shape,str(Path(info['raw_path']).resolve()))
    expected=block_average(full['planck'][centered_raw_window(raw.shape,8,images['factor'])],images['factor'])
    first=aligned_images(info,images,8,sky)
    np.testing.assert_array_equal(first['planck'],expected)
    header=fits.getheader(first['cache_path'])
    assert header['BUNIT']=='y' and not header['NORMALIZ'] and not header['AUGMENT']
    with (patch('clusterprep.io.load_celestial_geometry',side_effect=AssertionError('radio read')),
         patch('clusterprep.planck.project',side_effect=AssertionError('full map read'))):
        second=aligned_images(info,images,8,sky)
    assert second['cache_status']=='reused'
    np.testing.assert_array_equal(second['planck'],expected)
    Path(first['cache_path']).write_bytes(b'broken')
    assert aligned_images(info,images,8,sky)['cache_status']=='rebuilt'


def test_memory_reuse_copies_and_invalidation(sky,sources):
    dataset=ds(sources,sky,['0'],cache=True,normalization={'planck':{'method':'none'}})
    original=dataset[0]
    pristine=original['planck'].clone()
    with patch.object(dataset,'_load_modality_images',side_effect=AssertionError('cache miss')):
        original['planck'].fill_(100)
        assert not torch.all(dataset[0]['planck']==100)
        images=dataset._modality_images(dataset.data_info['0'])
        images['planck'][:]=200
        assert not torch.all(dataset[0]['planck']==200)
    assert len(dataset._modality_cache)==1
    with fits.open(sky['source_path'],mode='update') as hdus:
        hdus[1].data['FULL'][:]*=2
    updated=dataset[0]
    torch.testing.assert_close(updated['planck'],pristine*2)
    path=dataset._modality_images(dataset.data_info['0'])['cache_path']
    Path(path).write_bytes(b'corrupt reduced')
    with patch.object(dataset,'_load_modality_images',wraps=dataset._load_modality_images) as load:
        dataset[0]
        assert load.call_count==1
    assert len(dataset._modality_cache)==1


def test_radio_change_and_size_invalidate_disk(sky,sources):
    info=sources['0']
    a=aligned_images(info,preprocess(info,'raw',8),8,sky)
    b=aligned_images(info,preprocess(info,'raw',16),16,sky)
    assert a['cache_path']!=b['cache_path']
    with fits.open(info['raw_path'],mode='update') as hdus:
        hdus[0].header['CRVAL1']+=5
    c=aligned_images(info,preprocess(info,'raw',8),8,sky)
    assert c['cache_path']!=a['cache_path']


def test_cache_controls(sky,sources):
    for memory,force in [(False,False),(True,True)]:
        dataset=ds(sources,{**sky,'force_rebuild':force},['0'],cache=memory,
                   normalization={'planck':{'method':'none'}})
        with patch.object(dataset,'_load_modality_images',wraps=dataset._load_modality_images) as load:
            dataset[0];dataset[0]
            assert load.call_count==2


def test_tensor_equivalence_all_variants(sky,sources):
    options=dict(training=True,augment=True,modalities=['radio','xray','planck'],
                 zoom=dict(enabled=True,factors=[1.,1.5,.8]),normalization={'planck':{'method':'none'}})
    cached=ds(sources,sky,['1'],cache=True,**options)
    uncached=ds(sources,sky,['1'],cache=False,**options)
    for i in range(len(cached)):
        a,b=cached[i],uncached[i]
        for name in ['image','availability','raw','chandra','planck','planck_mask']:
            torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
    assert len(cached._modality_cache)==1


def test_normalization_not_cached(sky,sources):
    dataset=ds(sources,sky,['0'],cache=True,normalization_bounds={'planck':[[0.,1.]]})
    a=dataset[0]['planck']
    dataset.normalization_bounds={'planck':[[0.,2.]]}
    with patch.object(dataset,'_load_modality_images',side_effect=AssertionError('unnecessary reread')):
        torch.testing.assert_close(dataset[0]['planck'],a/2)


def test_noise_not_frozen(sky,sources):
    dataset=ds(sources,sky,['0'],cache=True,training=True,normalization={'planck':{'method':'none'}})
    dataset.noise_enabled=True
    calls=[]
    def noise(image,*args):
        calls.append(1)
        result=image.copy();result[0,0]+=len(calls)*100
        return result
    with patch('clusterprep.dataset.prepare_noise_context',return_value={}),patch('clusterprep.dataset.add_radio_noise',side_effect=noise):
        dataset[0];dataset[0]
    assert len(calls)==2
    assert len(dataset._modality_cache)==1


def test_matched_avoids_unused_full_projection(sky,catalog):
    info,cid,root=catalog
    dataset=ClusterDataset(info,[cid],['RR','RH'],task='rr_rh',size=128,mode='multimodal',
        modalities=['radio','xray','planck'],planck=sky,cache=True,processed_dir=root,
        radio_preprocessing='matched_resolution',normalization={'planck':{'method':'none'}})
    with patch('clusterprep.planck.project',wraps=project) as call:
        dataset[0];dataset[0]
        assert call.call_count==1
        assert call.call_args.args[2]==(128,128)
        assert call.call_args.args[3].endswith(':matched')


@pytest.mark.parametrize('shape',[(32,32),(1,1,32,32),(2,1,32,32)])
def test_header_geometry_matches_reader(tmp_path,shape):
    w=WCS(naxis=2);w.wcs.ctype=['RA---TAN','DEC--TAN']
    path=tmp_path/'cube.fits'
    fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(np.zeros(shape,dtype='float32'),w.to_header())]).writeto(path)
    image,_,old_wcs=load_celestial_fits(path)
    with patch.object(fits.ImageHDU,'data',property(lambda self: (_ for _ in ()).throw(AssertionError('pixels loaded')))):
        new_shape,_,new_wcs=load_celestial_geometry(path)
    assert new_shape==image.shape
    assert new_wcs.to_header()==old_wcs.to_header()
