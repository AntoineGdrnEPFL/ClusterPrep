"""Read-only checks of the actual radio-noise kernel and preprocessing stages."""
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve

from .radio_noise import _kernel, background_mask
from .preprocessing import normalize


def lag_one_correlations(image):
    """Pearson correlations over finite neighbouring pairs, x=columns/y=rows."""
    image = np.asarray(image, dtype=float)
    result = {}
    for name, a, b in [('x', image[:, :-1], image[:, 1:]),
                       ('y', image[:-1, :], image[1:, :])]:
        valid = np.isfinite(a) & np.isfinite(b)
        av, bv = a[valid], b[valid]
        result[name] = (float(np.corrcoef(av, bv)[0, 1])
                        if av.size >= 3 and av.std() > 0 and bv.std() > 0 else np.nan)
        result['pairs_' + name] = int(av.size)
    return result


def autocorrelation_window(image, radius=20):
    """Pair-count corrected covariance / lag-zero covariance on a centred grid.

    Uses a global finite-pixel mean; unlike the lag-one Pearson statistic, this
    finite-sample estimator is not strictly bounded by one at nonzero lags.
    A zero-variance input is explicitly undefined (NaN).
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not isinstance(radius, int) or not 0 <= radius < min(image.shape):
        raise ValueError('Image 2D et rayon entier inférieur aux dimensions requis')
    valid = np.isfinite(image)
    if valid.sum() < 3:
        return np.full((2*radius+1, 2*radius+1), np.nan)
    centered = np.where(valid, image-image[valid].mean(), 0)
    numerator = fftconvolve(centered, centered[::-1, ::-1], mode='full')
    counts = fftconvolve(valid.astype(float), valid[::-1, ::-1].astype(float), mode='full')
    ac = np.divide(numerator, counts, out=np.full_like(numerator, np.nan), where=counts > .5)
    cy, cx = np.array(image.shape)-1
    variance = ac[cy, cx]
    if not np.isfinite(variance) or variance <= 0:
        return np.full((2*radius+1, 2*radius+1), np.nan)
    return ac[cy-radius:cy+radius+1, cx-radius:cx+radius+1] / variance


def kernel_autocorrelation(context, radius=20):
    """Infinite-field expectation for the SAME sampled kernel used by the pipeline.

    In particular, do not use the continuous Gaussian approximation for beams
    narrower than a pixel. Recentring/sample RMS introduce finite-image effects.
    """
    if not isinstance(radius, int) or radius < 0:
        raise ValueError('Rayon entier positif ou nul requis')
    kernel = _kernel(tuple(np.asarray(context['covariance']).ravel()))
    ac = fftconvolve(kernel, kernel[::-1, ::-1], mode='full')
    cy, cx = np.array(kernel.shape)-1
    ac /= ac[cy, cx]
    out = np.zeros((2*radius+1, 2*radius+1))
    ry, rx = min(radius, cy), min(radius, cx)
    out[radius-ry:radius+ry+1, radius-rx:radius+rx+1] = ac[cy-ry:cy+ry+1, cx-rx:cx+rx+1]
    return out


def beam_diagnostics(context, wcs, shape):
    """Check pixel covariance against independent sky endpoints through the WCS."""
    import astropy.units as u
    covariance = np.asarray(context['covariance'], dtype=float)
    vals, vecs = np.linalg.eigh(covariance)
    widths = np.sqrt(vals) * np.sqrt(8*np.log(2))
    x, y = (shape[1]-1)/2, (shape[0]-1)/2
    center = wcs.pixel_to_world(x, y)
    header = context['metadata']
    vectors = []
    for width, pa in [(header['BMAJ'], header['BPA']), (header['BMIN'], header['BPA']+90)]:
        plus = center.directional_offset_by(pa*u.deg, width/2*u.deg)
        minus = center.directional_offset_by((pa+180)*u.deg, width/2*u.deg)
        vectors.append(np.asarray(wcs.world_to_pixel(plus))-np.asarray(wcs.world_to_pixel(minus)))
    axes = np.column_stack(vectors)
    direct = axes @ axes.T / (8*np.log(2))
    # Principal axes on the pixel grid can differ from projected sky axes for
    # unequal pixel scales/shear. Compare covariance, not just axis lengths.
    relative_error = float(np.linalg.norm(direct-covariance)/np.linalg.norm(covariance))
    return dict(fwhm_minor_px=float(widths[0]), fwhm_major_px=float(widths[1]),
                angle_xy_deg=float(np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1])) % 180),
                endpoint_relative_error=relative_error,
                pixel_covariance=covariance.tolist(), endpoint_covariance=direct.tolist(),
                pixel_scales_arcsec=(np.linalg.norm(context['jacobian'], axis=0)*3600).tolist())


def background_diagnostics(image):
    """Clipping and sensitivity to a 1/2-pixel buffer around positive outliers.

    This detects sensitivity to bright-source wings; it is not a source catalogue
    or proof of clean sky. Low surface-brightness diffuse emission can remain.
    """
    image = np.asarray(image)
    kept = background_mask(image)
    y, x = np.indices(image.shape)
    h, w = image.shape
    candidate = np.isfinite(image) & ((x < w/4) | (x >= 3*w/4) | (y < h/4) | (y >= 3*h/4))
    def mad(mask):
        values = image[mask].astype(float)
        if values.size < 32:
            return np.nan
        return float(1.4826022185*np.median(np.abs(values-np.median(values))))
    median = float(np.median(image[kept]))
    positive = candidate & ~kept & (image > median)
    buffers = {n: kept & ~binary_dilation(positive, iterations=n) for n in (1, 2)}
    return dict(candidate=candidate, kept=kept, rejected=candidate & ~kept,
                positive_rejected=positive, median=median,
                rms_before=mad(candidate), rms_after=mad(kept),
                rms_buffer1=mad(buffers[1]), rms_buffer2=mad(buffers[2]),
                n_candidate=int(candidate.sum()), n_kept=int(kept.sum()),
                n_positive_rejected=int(positive.sum()),
                n_buffer1=int(buffers[1].sum()), n_buffer2=int(buffers[2].sum()))


def pipeline_residuals(original, augmented, angle=30, flip=False):
    """Measure perturbations at three stages, matching Dataset normalization/TF.

    The final two residuals are signal-dependent, clipped/asinh transformed,
    not stationary additive Gaussian noise. Nearest rotations resample pixels.
    """
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    valid = np.isfinite(original) & np.isfinite(augmented)
    bounds = np.quantile(np.asarray(original)[np.isfinite(original)].astype(float), [.30, .99])
    clean = normalize(original, bounds=bounds)
    noisy = normalize(augmented, bounds=bounds)
    tensor = torch.from_numpy(np.stack([clean, noisy, valid.astype(np.float32)]))
    if flip:
        tensor = TF.hflip(tensor)
    tensor = TF.rotate(tensor, angle=angle, interpolation=InterpolationMode.NEAREST, expand=False, fill=0)
    transformed = tensor.numpy()
    return dict(linear=np.where(valid, augmented-original, np.nan),
                normalized=np.where(valid, noisy-clean, np.nan),
                geometry=np.where(transformed[2] > .5, transformed[1]-transformed[0], np.nan))
