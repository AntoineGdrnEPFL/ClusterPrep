import numpy as np
import pytest
from astropy.wcs import WCS
from astropy.io import fits
from clusterprep.radio_psf import (Beam, RadioPlane, read_radio, jacobian, target_beam,
                                   kernel_covariance, gaussian_kernel, match_native,
                                   surface_brightness_factor, FWHM_SIGMA)
from clusterprep.matched_resolution import match_pair, prepare_matched_pair


def grid(size=256, scale=1., angle=0., anisotropy=1.):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [170., 30.]
    w.wcs.crpix = [(size+1)/2]*2
    a = np.deg2rad(angle)
    w.wcs.cd = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.diag([-scale, scale*anisotropy])/3600
    w.array_shape = (size, size)
    return w


def source(beam, wcs, size=256, flux=1., offset=(0, 0)):
    yy, xx = np.indices((size, size)) - (size-1)/2
    coords = np.stack((xx-offset[0], yy-offset[1]), axis=-1)
    j = jacobian(wcs, (size-1)/2, (size-1)/2)
    cov = np.linalg.inv(j) @ beam.covariance @ np.linalg.inv(j).T
    data = flux*np.exp(-.5*np.einsum('...i,ij,...j->...', coords, np.linalg.inv(cov), coords))
    return RadioPlane(data, np.ones_like(data, bool), wcs, beam, 'Jy/beam')


def moments(data):
    data = np.nan_to_num(data).astype(float)
    yy, xx = np.indices(data.shape)
    x, y = np.sum(data*xx)/data.sum(), np.sum(data*yy)/data.sum()
    return np.array([[np.sum(data*(xx-x)**2), np.sum(data*(xx-x)*(yy-y))],
                     [np.sum(data*(xx-x)*(yy-y)), np.sum(data*(yy-y)**2)]])/data.sum()


@pytest.mark.parametrize('angle,anisotropy', [(0., 1.), (37., 1.7)])
def test_flux_psf_and_shared_chandra(angle, anisotropy):
    native1 = grid(angle=angle, anisotropy=anisotropy)
    native2 = grid(angle=-21., anisotropy=1.2)
    raw = source(Beam(6, 3, 32), native1)
    tapered = source(Beam(9, 5, 117), native2)
    out = grid(128, 1.5, angle=13, anisotropy=1.3)
    xray = np.arange(128**2, dtype=np.float32).reshape(128, 128)
    xm = np.ones_like(xray, bool); xm[:4] = False
    pair = match_pair(raw, tapered, out, xray, xm)
    assert np.array_equal(pair.chandra, xray)
    assert np.array_equal(pair.chandra_mask, xm)
    target, minimum = target_beam((raw, tapered), out)
    assert minimum >= 3-1e-8
    j = jacobian(out, 63.5, 63.5)
    expected = np.linalg.inv(j) @ target.covariance @ np.linalg.inv(j).T
    for image, mask in [(pair.raw, pair.raw_mask), (pair.t50kcpsub, pair.t50kcpsub_mask)]:
        assert image.shape == (128, 128)
        assert np.array_equal(mask, np.isfinite(image))
        # Bilinear interpolation adds at most a small native-pixel covariance.
        fwhm = FWHM_SIGMA*np.sqrt(np.linalg.eigvalsh(moments(image)))
        np.testing.assert_allclose(fwhm, FWHM_SIGMA*np.sqrt(np.linalg.eigvalsh(expected)), rtol=.035)
        flux = np.nansum(image)*abs(np.linalg.det(j))*(np.pi/180/3600)**2/target.solid_angle_sr
        assert flux == pytest.approx(1, rel=.015)
    tensors = pair.tensors()
    assert tensors['raw'].shape == (1, 128, 128)
    assert np.isfinite(tensors['raw'].numpy()).all()
    again = match_pair(raw, tapered, out, xray, xm)
    np.testing.assert_array_equal(pair.raw, again.raw)


def test_masks_and_surface_brightness():
    plane = RadioPlane(np.ones((256, 256))*2, np.ones((256, 256), bool), grid(), Beam(6, 4, 50), 'mJy/beam')
    plane.mask[120:136, 120:136] = False
    target = Beam(10, 10, 0)
    data, valid, support, _ = match_native(plane, target)
    assert not valid[128, 128] and not valid[0].any()
    assert np.isnan(data[128, 128])
    np.testing.assert_allclose(data[valid], 2*surface_brightness_factor('mJy/beam', plane.beam), rtol=1e-12)
    assert support[128, 128] < .5
    plane.mask[:] = False
    data, valid, _, _ = match_native(plane, target)
    assert not valid.any() and np.isnan(data).all()


def test_beam_convention_psd_and_singular_kernel():
    beam = Beam(8, 4, 0)
    assert beam.covariance[1, 1] > beam.covariance[0, 0]  # major axis north
    with pytest.raises(ValueError, match='deconvolution'):
        kernel_covariance(beam, Beam(5, 5, 0), np.eye(2))
    np.testing.assert_array_equal(gaussian_kernel(np.zeros((2, 2))), [[1]])
    kernel = gaussian_kernel(np.array([[4., 4.], [4., 4.]]))
    assert np.isfinite(kernel).all() and kernel.min() >= 0
    assert kernel.sum() == pytest.approx(1)
    np.testing.assert_array_equal(kernel, gaussian_kernel(np.array([[4., 4.], [4., 4.]])))


@pytest.mark.parametrize('unit', ['', 'K', 'Jy/pixel'])
def test_reject_ambiguous_units(unit):
    with pytest.raises(ValueError):
        surface_brightness_factor(unit, Beam(8, 4, 0))


def test_native_reader_and_physical_field(tmp_path):
    w = grid()
    paths = []
    for i, beam in enumerate([Beam(6, 3, 32), Beam(9, 5, 117)]):
        plane = source(beam, w)
        header = w.to_header()
        header.update(BMAJ=beam.major_arcsec/3600, BMIN=beam.minor_arcsec/3600, BPA=beam.pa_deg, BUNIT='JY/BEAM')
        path = tmp_path/f'{i}.fits'; fits.writeto(path, plane.data[None, None], header)
        paths.append(path)
    xp = tmp_path/'xray.fits'; fits.writeto(xp, np.ones((256, 256)), w.to_header())
    record = dict(cluster_id='synthetic', ra_deg=170., dec_deg=30., redshift=.2, r500_kpc=100)
    pair = prepare_matched_pair(*paths, xp, record)
    assert pair.raw.shape == (128, 128)
    assert pair.metadata['field']['fov_method'] == 'r500'
    assert pair.wcs.pixel_to_world(63.5, 63.5).ra.deg == pytest.approx(170.)
    with pytest.raises(ValueError, match='4R500'):
        prepare_matched_pair(*paths, xp, {**record, 'redshift': None})
    header = w.to_header(); header.update(BMAJ=.01, BMIN=.005, BPA=0, BUNIT='Jy/beam')
    cube = tmp_path/'cube.fits'; fits.writeto(cube, np.zeros((2, 256, 256)), header)
    with pytest.raises(ValueError, match='cube'):
        read_radio(cube)


def test_diffuse_relic_and_noise():
    from scipy.signal import fftconvolve
    size = 256
    w = grid(size)
    yy, xx = np.indices((size, size)) - (size-1)/2
    diffuse = np.exp(-(xx**2+yy**2)/(2*18**2))
    relic = .8*np.exp(-((xx+35)**2/(2*5**2)+(yy-20)**2/(2*24**2)))
    sky = diffuse + relic
    rng = np.random.default_rng(19)
    planes = []
    for beam in (Beam(6, 3, 40), Beam(9, 5, 113)):
        j = jacobian(w, 127.5, 127.5)
        cov = np.linalg.inv(j) @ beam.covariance @ np.linalg.inv(j).T
        image = fftconvolve(sky, gaussian_kernel(cov), mode='same')
        image += rng.normal(0, 1e-4, image.shape)
        planes.append(RadioPlane(image, np.ones_like(image, bool), w, beam, 'Jy/sr'))
    pair = match_pair(*planes, grid(128, 1.5), np.zeros((128, 128)), np.ones((128, 128), bool), output_unit='Jy/sr')
    valid = pair.common_radio_mask
    assert np.sqrt(np.mean((pair.raw[valid]-pair.t50kcpsub[valid])**2)) < 1e-4


def test_nonstationary_wcs_is_rejected():
    from clusterprep.radio_psf import grid_jacobians
    with pytest.raises(ValueError, match='Nonstationary'):
        grid_jacobians(grid(scale=1000), (256, 256))


def test_anisotropic_sampling_and_shifted_hole():
    raw = source(Beam(6, 3, 32), grid())
    tapered = source(Beam(9, 5, 117), grid(angle=10))
    raw.mask[100:120, 100:120] = False
    output = grid(128, 2, anisotropy=2, angle=27)
    target, minimum = target_beam((raw, tapered), output, pixels_per_beam=4)
    assert target.major_arcsec == pytest.approx(16, rel=1e-4)
    assert minimum == pytest.approx(4, rel=1e-4)
    pair = match_pair(raw, tapered, output, np.zeros((128, 128)), np.ones((128, 128), bool), pixels_per_beam=4)
    assert not pair.raw_mask.all()
    assert np.isnan(pair.raw[~pair.raw_mask]).all()
    assert pair.metadata['nyquist_amplitude'] < 1e-6
