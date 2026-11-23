"""Independent radio PSF matching; angular covariances use east/north arcsec²."""
from dataclasses import dataclass

import numpy as np
import astropy.units as u
from astropy.io import fits
from scipy.signal import fftconvolve
from scipy.integrate import quad
from scipy.special import ndtr
from astropy.convolution import Gaussian2DKernel

from .io import _find_celestial_image_hdu

FWHM_SIGMA = np.sqrt(8 * np.log(2))


@dataclass(frozen=True)
class Beam:
    major_arcsec: float
    minor_arcsec: float
    pa_deg: float

    def __post_init__(self):
        if (not np.isfinite([self.major_arcsec, self.minor_arcsec, self.pa_deg]).all()
                or not self.major_arcsec >= self.minor_arcsec > 0):
            raise ValueError('Beam requires finite BMAJ >= BMIN > 0 and BPA')

    @classmethod
    def from_header(cls, header):
        try:
            return cls(float(header['BMAJ']) * 3600, float(header['BMIN']) * 3600,
                       float(header['BPA']))
        except (KeyError, TypeError) as exc:
            raise ValueError('Missing/non-numeric BMAJ, BMIN or BPA') from exc

    @property
    def covariance(self):
        angle = np.deg2rad(self.pa_deg)
        axes = np.array([[np.sin(angle), np.cos(angle)],
                         [np.cos(angle), -np.sin(angle)]])
        return axes @ np.diag((np.array([self.major_arcsec, self.minor_arcsec]) /
                               FWHM_SIGMA) ** 2) @ axes.T

    @property
    def solid_angle_sr(self):
        return np.pi / (4 * np.log(2)) * self.major_arcsec * self.minor_arcsec * u.arcsec.to(u.rad)**2


def jacobian(wcs, x, y):
    """Finite-difference pixel -> local east/north, including PC/CD and projection."""
    center = wcs.pixel_to_world(x, y)
    columns = []
    for dx, dy in ((.5, 0), (0, .5)):
        lo = center.spherical_offsets_to(wcs.pixel_to_world(x-dx, y-dy))
        hi = center.spherical_offsets_to(wcs.pixel_to_world(x+dx, y+dy))
        columns.append([(b-a).to_value(u.arcsec) for a, b in zip(lo, hi)])
    result = np.array(columns).T
    if not np.isfinite(result).all() or abs(np.linalg.det(result)) < 1e-12:
        raise ValueError('Invalid/singular celestial Jacobian')
    return result


def grid_jacobians(wcs, shape, tolerance=.01):
    """Reject fields where a stationary native convolution is not justified.

    Check a 5x5 mesh including pixel boundaries. Tolerance is relative matrix
    variation (default 1%); this is a local approximation, not a distortion solver.
    """
    if not wcs.has_celestial or wcs.pixel_n_dim != 2:
        raise ValueError('A two-dimensional celestial WCS is required')
    h, width = shape
    center = jacobian(wcs, (width-1)/2, (h-1)/2)
    mesh = np.array([jacobian(wcs, x, y) for y in np.linspace(-.5, h-.5, 5)
                     for x in np.linspace(-.5, width-.5, 5)])
    variation = max(np.linalg.norm(j @ np.linalg.inv(center)-np.eye(2), 2) for j in mesh)
    if variation > tolerance:
        raise ValueError(f'Nonstationary WCS ({variation:.3g} > {tolerance}); local kernels required')
    return center, mesh, float(variation)


def surface_brightness_factor(bunit, beam):
    """Explicit multiplier from FITS units to Jy/sr; no K or Jy/pixel guessing."""
    if not isinstance(bunit, str) or not bunit.strip():
        raise ValueError('Missing BUNIT')
    unit = u.Unit(bunit.strip().replace('JY', 'Jy').replace('BEAM', 'beam'))
    if unit.is_equivalent(u.Jy/u.beam):
        return unit.to(u.Jy/u.beam) / beam.solid_angle_sr
    if unit.is_equivalent(u.Jy/u.sr):
        return unit.to(u.Jy/u.sr)
    raise ValueError(f'Unsupported radio BUNIT: {bunit}')


@dataclass
class RadioPlane:
    data: np.ndarray
    mask: np.ndarray
    wcs: object
    beam: Beam
    bunit: str
    source: str = ''

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 2 or min(self.data.shape) < 2:
            raise ValueError('Radio image must be 2D with both dimensions >= 2')
        if np.shape(self.mask) != self.data.shape:
            raise ValueError('Coverage mask shape differs from image')
        self.mask = np.asarray(self.mask, dtype=bool) & np.isfinite(self.data)
        surface_brightness_factor(self.bunit, self.beam)


def read_radio(path, mask=None):
    """Read native metadata, allowing singleton continuum axes only.

    External coverage maps must be supplied explicitly; finite zero is observed
    data, never inferred to mean a missing observation.
    """
    with fits.open(path, memmap=False) as hdul:
        hdu, wcs = _find_celestial_image_hdu(hdul)
        data = np.squeeze(hdu.data).astype(float)
        if data.ndim != 2:
            raise ValueError('Non-singleton FITS cube: select a physical plane explicitly')
        return RadioPlane(data, np.isfinite(data) if mask is None else mask, wcs,
                          Beam.from_header(hdu.header), hdu.header.get('BUNIT'), str(path))


def target_beam(planes, target_wcs, shape=(128, 128), pixels_per_beam=3.):
    if not np.isfinite(pixels_per_beam) or pixels_per_beam <= 0:
        raise ValueError('pixels_per_beam must be positive')
    if len(planes) != 2:
        raise ValueError('Exactly two radio products are required')
    _, mesh, _ = grid_jacobians(target_wcs, shape)
    # Largest singular value is the conservative scale for a circular sky beam.
    scale = max(np.linalg.svd(j, compute_uv=False)[0] for j in mesh)
    width = max(*(p.beam.major_arcsec for p in planes), pixels_per_beam * scale)
    target = Beam(width, width, 0.)
    minimum = min(FWHM_SIGMA * np.sqrt(np.linalg.eigvalsh(
        np.linalg.inv(j) @ target.covariance @ np.linalg.inv(j).T)[0]) for j in mesh)
    return target, float(minimum)


def kernel_covariance(native, target, jac):
    difference = target.covariance - native.covariance
    values, vectors = np.linalg.eigh(difference)
    if values.min() < -1e-10 * np.linalg.norm(target.covariance, 2):
        raise ValueError('Target beam requires deconvolution (non-PSD kernel)')
    difference = (vectors * np.maximum(values, 0)) @ vectors.T
    inverse = np.linalg.inv(jac)
    return inverse @ difference @ inverse.T


def gaussian_kernel(covariance, truncate=5., max_side=2049):
    """Normalized finite Gaussian; singular kernels integrated over pixel cells.

    Broad kernels are sampled at pixel centres. Narrow/singular kernels use
    Gaussian cell probabilities and thus have a small discretization error.
    The caller records and checks the actual discrete covariance.
    """
    values, vectors = np.linalg.eigh(covariance)
    if values.min() < -1e-10:
        raise ValueError('Non-PSD pixel kernel')
    if values.max() < 1e-12:
        return np.ones((1, 1))
    if not np.isfinite(truncate) or truncate < 4:
        raise ValueError('truncate must be >= 4 sigma')
    radius = int(np.ceil(truncate * np.sqrt(values.max())))
    if 2*radius+1 > max_side:
        raise ValueError('Kernel exceeds max_side; explicit larger resource budget needed')
    if values.min() >= .75**2:
        theta = np.arctan2(vectors[1, 1], vectors[0, 1])
        kernel = Gaussian2DKernel(np.sqrt(values[1]), np.sqrt(values[0]), theta,
                                  x_size=2*radius+1, y_size=2*radius+1).array
    else:
        y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
        points = np.stack((x, y), axis=-1)
        kernel = np.zeros(x.shape, dtype=float)
        if values[0] < 1e-12 * values[1]:
            # Rank-one Gaussian: intersection of the cell's two intervals in z.
            direction = vectors[:, 1] * np.sqrt(values[1])
            lo, hi = np.full(x.shape, -np.inf), np.full(x.shape, np.inf)
            for axis, step in enumerate(direction):
                if abs(step) < 1e-14:
                    hi = np.where(np.abs(points[..., axis]) <= .5, hi, -np.inf)
                else:
                    a = (points[..., axis]-.5)/step
                    b = (points[..., axis]+.5)/step
                    lo, hi = np.maximum(lo, np.minimum(a, b)), np.minimum(hi, np.maximum(a, b))
            kernel = np.maximum(ndtr(hi)-ndtr(lo), 0)
        else:
            # Conditional normal integral; deterministic and compatible with the
            # project's existing SciPy dependency, including very narrow axes.
            sx = np.sqrt(covariance[0, 0])
            slope = covariance[1, 0]/sx
            sy = np.sqrt(np.linalg.det(covariance)/covariance[0, 0])
            for index in np.ndindex(kernel.shape):
                px, py = points[index]
                def integrand(z):
                    return np.exp(-z*z/2)/np.sqrt(2*np.pi) * (
                        ndtr((py+.5-slope*z)/sy)-ndtr((py-.5-slope*z)/sy))
                kernel[index] = quad(integrand, (px-.5)/sx, (px+.5)/sx,
                                     epsabs=1e-12, epsrel=1e-10)[0]
    return kernel / kernel.sum()


def match_native(plane, target, *, min_support=.999, truncate=5., psf_tolerance=.03):
    """Convolve Jy/sr with valid-weight normalization; preserve native holes.

    Outside support is zero *weight*, not an observation. Pixels with insufficient
    kernel mass or no original observation remain NaN. Photometry is only promised
    away from these boundaries. Returns data, validity, support and diagnostics.
    """
    if not np.isfinite(min_support) or not 0 < min_support <= 1:
        raise ValueError('min_support must be in (0, 1]')
    if not np.isfinite(psf_tolerance) or psf_tolerance <= 0:
        raise ValueError('psf_tolerance must be positive')
    jac, _, variation = grid_jacobians(plane.wcs, plane.data.shape)
    covariance = kernel_covariance(plane.beam, target, jac)
    kernel = gaussian_kernel(covariance, truncate)
    yy, xx = np.indices(kernel.shape) - (np.array(kernel.shape)[:, None, None]-1)/2
    measured = np.array([[np.sum(kernel*xx*xx), np.sum(kernel*xx*yy)],
                         [np.sum(kernel*xx*yy), np.sum(kernel*yy*yy)]])
    effective = plane.beam.covariance + jac @ measured @ jac.T
    error = np.linalg.norm(effective-target.covariance, 2)/np.linalg.norm(target.covariance, 2)
    if error > psf_tolerance:
        raise ValueError(f'Discrete kernel PSF covariance error {error:.3g} exceeds tolerance')
    support = np.clip(fftconvolve(plane.mask.astype(float), kernel, mode='same'), 0, 1)
    brightness = np.where(plane.mask, plane.data, 0) * surface_brightness_factor(plane.bunit, plane.beam)
    numerator = fftconvolve(brightness, kernel, mode='same')
    mask = plane.mask & (support > 0) & (support >= min_support-1e-12)
    result = np.full(plane.data.shape, np.nan)
    np.divide(numerator, support, out=result, where=mask)
    metadata = dict(source=plane.source, bunit_native=plane.bunit,
                    native_beam=vars(plane.beam), native_wcs=dict(plane.wcs.to_header()),
                    native_pixel_scales_arcsec=np.linalg.norm(jac, axis=0).tolist(),
                    native_coverage=float(plane.mask.mean()), kernel_covariance_pixels=covariance.tolist(),
                    discrete_effective_covariance_arcsec2=effective.tolist(),
                    discrete_covariance_relative_error=float(error), wcs_variation=variation,
                    min_support=min_support, truncate_sigma=truncate)
    return result, mask, support, metadata
