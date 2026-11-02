"""Optional linear-radio noise, on the reduced grid before normalization.

The restoring-beam kernel is an approximation, not an interferometer simulator.
RMS is estimated in the same units and resolution as the reduced input.
"""
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
from scipy.signal import fftconvolve


DEFAULTS = dict(enabled=False, method="gaussian_pixel", probability=0.5,
                amplitude_range=[0.0, 0.5], rms_source="estimated",
                rms_estimation=dict(method="sigma_clipped_mad", region="background", fallback="raise"),
                missing_beam_policy="raise")


def validate_radio_noise(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("radio_noise: objet JSON ou champs inconnus")
    config = {**DEFAULTS, **value}
    if not isinstance(config['enabled'], bool):
        raise ValueError("radio_noise.enabled doit être booléen")
    if config['method'] not in {'gaussian_pixel', 'beam_correlated'}:
        raise ValueError("radio_noise.method invalide")
    p = config['probability']
    a = config['amplitude_range']
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("radio_noise.probability doit être dans [0,1]")
    if (not isinstance(a, (list, tuple)) or len(a) != 2 or
        any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) for x in a) or
        not 0 <= a[0] <= a[1]):
        raise ValueError("radio_noise.amplitude_range invalide")
    if config['rms_source'] != 'estimated':
        raise ValueError("Seul rms_source=estimated est disponible : aucun RMS catalogue validé")
    if config['rms_estimation'] != DEFAULTS['rms_estimation']:
        raise ValueError("RMS: sigma_clipped_mad/background/raise requis")
    if config['missing_beam_policy'] not in {'raise', 'skip', 'gaussian_pixel'}:
        raise ValueError("missing_beam_policy invalide")
    return config


def background_mask(image):
    """Outer quarter on each edge, then iterative 3-sigma MAD rejection.

    Reject compact sources robustly; extensive diffuse emission may still bias
    this geometric background proxy. At least 32 retained pixels are required.
    """
    image = np.asarray(image)
    y, x = np.indices(image.shape)
    h, w = image.shape
    mask = np.isfinite(image) & ((x < w / 4) | (x >= 3*w/4) | (y < h/4) | (y >= 3*h/4))
    for _ in range(8):
        values = image[mask].astype(float)
        if values.size < 32:
            raise ValueError("Fond radio insuffisant (<32 pixels)")
        median = np.median(values)
        rms = 1.4826022185 * np.median(np.abs(values - median))
        if not np.isfinite(rms) or rms <= 0:
            raise ValueError("RMS radio nul/non fini : fond inutilisable")
        new = mask & (np.abs(image - median) <= 3*rms)
        if np.array_equal(new, mask):
            return mask
        mask = new
    if mask.sum() < 32:
        raise ValueError("Fond radio insuffisant après clipping")
    return mask


def estimate_rms(image):
    values = np.asarray(image)[background_mask(image)].astype(float)
    return float(1.4826022185 * np.median(np.abs(values - np.median(values))))


def local_jacobian(wcs, shape):
    """Pixel (x,y) -> local east,north tangent offsets in degrees."""
    x, y = (shape[1]-1)/2, (shape[0]-1)/2
    center = wcs.pixel_to_world(x, y)
    columns = []
    for dx, dy in ((0.5, 0), (0, 0.5)):
        lo = center.spherical_offsets_to(wcs.pixel_to_world(x-dx, y-dy))
        hi = center.spherical_offsets_to(wcs.pixel_to_world(x+dx, y+dy))
        columns.append([hi[i].deg - lo[i].deg for i in range(2)])
    jac = np.array(columns).T
    if not np.isfinite(jac).all() or abs(np.linalg.det(jac)) < 1e-20:
        raise ValueError("Jacobian WCS singulier/invalide")
    return jac


def beam_covariance(metadata, wcs, shape):
    try:
        major, minor, angle = (float(metadata[k]) for k in ('BMAJ', 'BMIN', 'BPA'))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BMAJ/BMIN/BPA absents ou invalides") from exc
    if not np.isfinite([major, minor, angle]).all() or not major >= minor > 0:
        raise ValueError("Beam: BMAJ >= BMIN > 0 requis (degrés)")
    theta = np.deg2rad(angle)
    axes = np.array([[np.sin(theta), np.cos(theta)], [np.cos(theta), -np.sin(theta)]])
    sky = axes @ np.diag(np.square([major, minor]) / (8*np.log(2))) @ axes.T
    inverse = np.linalg.inv(local_jacobian(wcs, shape))
    return inverse @ sky @ inverse.T


@lru_cache(maxsize=256)
def _kernel(covariance):
    cov = np.array(covariance).reshape(2, 2)
    if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError("Covariance beam invalide")
    rx, ry = np.maximum(1, np.ceil(4*np.sqrt(np.diag(cov))).astype(int))
    if max(rx, ry) > 2048:
        raise ValueError("Beam excessif sur la grille réduite")
    y, x = np.mgrid[-ry:ry+1, -rx:rx+1]
    coords = np.stack([x, y], axis=-1)
    kernel = np.exp(-0.5*np.einsum('...i,ij,...j->...', coords, np.linalg.inv(cov), coords))
    kernel /= kernel.sum()
    kernel.setflags(write=False)
    return kernel


def prepare_noise_context(images, info=None, cache_dir=None):
    """Cache RMS/beam by source provenance AND reduced grid; never cache draws."""
    path = None
    identity = None
    if info is not None and cache_dir is not None:
        import hashlib
        from .processed import _identity
        identity = _identity(info, 'raw', images['raw'].shape[0])
        token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = Path(cache_dir) / 'radio_noise' / f'{token}.json'
        if path.exists():
            try:
                saved = json.loads(path.read_text())
                if saved['version'] == 2 and saved['identity'] == identity:
                    return saved['context']
            except (ValueError, KeyError, OSError):
                pass
    metadata = dict(images.get('radio_metadata') or {})
    metadata_keys = ('BMAJ', 'BMIN', 'BPA', 'BUNIT')
    # Older preprocess functions/caches may omit all or part of the header.
    # Recover only from the actual source HDU, retaining the reduced-grid WCS.
    if info is not None and any(k not in metadata for k in metadata_keys):
        from .io import load_celestial_fits
        _, header, _ = load_celestial_fits(info['raw_path'])
        metadata = {**{k: header[k] for k in metadata_keys if k in header}, **metadata}
    context = dict(metadata=metadata, rms=estimate_rms(images['raw']),
                   jacobian=local_jacobian(images['wcs'], images['raw'].shape).tolist())
    try:
        context['covariance'] = beam_covariance(metadata, images['wcs'], images['raw'].shape).tolist()
    except ValueError as exc:
        context['beam_error'] = str(exc)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=path.parent, suffix='.json')
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(dict(version=2, identity=identity, context=context), stream, allow_nan=False)
            os.replace(temp, path)
        finally:
            Path(temp).unlink(missing_ok=True)
    return context


def add_radio_noise(image, context, config, rng=None):
    """Return float32 copy, retaining invalid pixels; RNG supports NumPy workers."""
    config = validate_radio_noise(config)
    out = np.asarray(image, dtype=np.float32).copy()
    rng = np.random if rng is None else rng
    if not config['enabled'] or config['probability'] == 0 or rng.random() >= config['probability']:
        return out
    alpha = rng.uniform(*config['amplitude_range'])
    if alpha == 0:
        return out
    method = config['method']
    if method == 'beam_correlated' and 'covariance' not in context:
        reason = context.get('beam_error', 'Métadonnées beam absentes')
        policy = config['missing_beam_policy']
        if policy == 'raise':
            raise ValueError(reason)
        warnings.warn(f'{reason}; missing_beam_policy={policy}', RuntimeWarning, stacklevel=2)
        if policy == 'skip':
            return out
        method = 'gaussian_pixel'
    valid = np.isfinite(out)
    rms = context['rms']
    if not np.isfinite(rms) or rms <= 0 or valid.sum() < 2:
        raise ValueError('RMS ou pixels valides insuffisants')
    if method == 'gaussian_pixel':
        noise = rng.standard_normal(out.shape)
    else:
        kernel = _kernel(tuple(np.asarray(context['covariance']).ravel()))
        padded = tuple(n+k-1 for n, k in zip(out.shape, kernel.shape))
        noise = fftconvolve(rng.standard_normal(padded), kernel, mode='valid')
        noise -= noise[valid].mean()
        scale = noise[valid].std()
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Champ corrélé dégénéré')
        noise /= scale
    out[valid] += (alpha*rms*noise[valid]).astype(np.float32)
    return out
