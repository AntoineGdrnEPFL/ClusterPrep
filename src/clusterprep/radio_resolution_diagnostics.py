"""Read-only diagnostic of native radio beams/noise on an existing target grid.

No reprojection, model changes, normalization for statistics, or training-cache
writes. Standalone JSON configuration, content-validated per-modality checkpoints.
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import math
import warnings

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from tqdm.auto import tqdm

from .io import load_celestial_fits
from .preprocessing import normalize
from .radio_noise import background_mask, local_jacobian, beam_covariance
from .common_wcs_validation import fingerprint, _atomic, _json

PRODUCTS = ('RAW', 'T50kpcSUB')
CLASSES = ('severe_lt1', 'probable_1to2', 'limited_2to3', 'comfortable_ge3')
DEFAULTS = dict(validation_dir='outputs/common_wcs_validation',
                output_dir='outputs/radio_resolution_diagnostic', cluster_ids=None,
                coverage_threshold=.8, coverage_difference_threshold=.1, dpi=180,
                overwrite=False)


def read_config(path, project_root):
    supplied = json.loads(Path(path).read_text())
    if set(supplied) - DEFAULTS.keys():
        raise ValueError(f'Unknown configuration fields: {set(supplied) - DEFAULTS.keys()}')
    config = {**DEFAULTS, **supplied}
    for key in ('coverage_threshold', 'coverage_difference_threshold'):
        if not isinstance(config[key], (int, float)) or not 0 <= config[key] <= 1:
            raise ValueError(f'{key} must be between 0 and 1')
    if not isinstance(config['dpi'], int) or config['dpi'] < 30:
        raise ValueError('dpi must be an integer >= 30')
    if not isinstance(config['overwrite'], bool):
        raise ValueError('overwrite must be boolean')
    ids = config['cluster_ids']
    if ids is not None and not (isinstance(ids, list) and all(isinstance(i, str) for i in ids)):
        raise ValueError('cluster_ids must be null or a list of identifiers')
    for key in ('validation_dir', 'output_dir'):
        config[key] = str((Path(project_root) / config[key]).resolve())
    return config


def sampling_class(pixels):
    if pixels is None or not np.isfinite(pixels) or pixels <= 0:
        return 'unknown'
    return CLASSES[0 if pixels < 1 else 1 if pixels < 2 else 2 if pixels < 3 else 3]


def canonical_unit(text):
    """Convert only explicit compatible units, never silently change beam area.

    FITS commonly uses JY/BEAM. Prefix case remains meaningful (mJy != MJy).
    Returns a multiplier and canonical unit, or raises for ambiguous units.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Missing BUNIT')
    normalized = text.strip().replace('JY', 'Jy').replace('BEAM', 'beam')
    aliases = {'jy/beam': 'Jy/beam', 'mjy/beam': 'mJy/beam', 'ujy/beam': 'uJy/beam'}
    normalized = aliases.get(normalized, normalized)
    unit = u.Unit(normalized)
    for target in (u.Jy/u.beam, u.Jy/u.sr, u.Jy/u.pix, u.K):
        if unit.is_equivalent(target):
            return float(unit.to(target)), target.to_string()
    raise ValueError(f'Unsupported intensity unit: {text!r}')


def _sky_area(wcs, shape):
    """Sum spherical quadrilaterals as two triangles; result in arcsec²."""
    y, x = np.mgrid[:shape[0]+1, :shape[1]+1]
    sky = wcs.pixel_to_world(x-.5, y-.5)
    vectors = np.moveaxis(sky.cartesian.xyz.value, 0, -1)
    if not np.isfinite(vectors).all():
        raise ValueError('Nonfinite sky coordinates at target pixel corners')
    def triangle(a, b, c):
        numerator = np.abs(np.einsum('...i,...i->...', a, np.cross(b, c)))
        denominator = 1 + np.sum(a*b, axis=-1) + np.sum(b*c, axis=-1) + np.sum(c*a, axis=-1)
        return 2 * np.arctan2(numerator, denominator)
    a, b, c, d = vectors[:-1, :-1], vectors[:-1, 1:], vectors[1:, 1:], vectors[1:, :-1]
    return float(np.sum(triangle(a, b, c)+triangle(a, c, d)) * u.sr.to(u.arcsec**2))


def grid_metrics(wcs, shape):
    jac = local_jacobian(wcs, shape) * 3600
    scales = np.linalg.norm(jac, axis=0)
    singular = np.linalg.svd(jac, compute_uv=False)
    area_pixel = float(abs(np.linalg.det(jac)))
    area = _sky_area(wcs, shape)
    cx, cy = (shape[1]-1)/2, (shape[0]-1)/2
    width = wcs.pixel_to_world(-.5, cy).separation(wcs.pixel_to_world(shape[1]-.5, cy)).arcsec
    height = wcs.pixel_to_world(cx, -.5).separation(wcs.pixel_to_world(cx, shape[0]-.5)).arcsec
    return dict(output_height=int(shape[0]), output_width=int(shape[1]),
                pixel_scale_x_arcsec=float(scales[0]), pixel_scale_y_arcsec=float(scales[1]),
                pixel_scale_arcsec=math.sqrt(area_pixel), pixel_scale_max_arcsec=float(singular.max()),
                pixel_anisotropy=float(singular.max()/singular.min()), pixel_area_arcsec2=area_pixel,
                fov_width_arcsec=float(width), fov_height_arcsec=float(height), fov_area_arcsec2=area,
                area_vs_central_jacobian=area/(area_pixel*np.prod(shape)))


def beam_metrics(header, wcs, shape, grid=None):
    """FWHM ellipse in pixel coordinates, plus explicitly separate area counts.

    Eigenvalues handle rotation, PC/CD, skew and anisotropic sampling. Directional
    lengths are also retained: sky BMAJ/BMIN vectors mapped by inverse Jacobian.
    The class uses the shortest principal FWHM in the pixel-coordinate ellipse.
    """
    grid = grid_metrics(wcs, shape) if grid is None else grid
    try:
        major, minor, angle = [float(header[k]) for k in ('BMAJ', 'BMIN', 'BPA')]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Missing/non-numeric BMAJ, BMIN or BPA') from exc
    if not np.isfinite([major, minor, angle]).all() or not major >= minor > 0:
        raise ValueError('Beam requires finite BMAJ >= BMIN > 0 and BPA')
    major *= 3600; minor *= 3600
    area = np.pi/(4*np.log(2)) * major * minor
    covariance = beam_covariance(header, wcs, shape)
    fwhm = np.sqrt(np.linalg.eigvalsh(covariance) * 8*np.log(2))
    inverse = np.linalg.inv(local_jacobian(wcs, shape)*3600)
    angle_rad = np.deg2rad(angle)
    major_direction = major * np.linalg.norm(inverse @ [np.sin(angle_rad), np.cos(angle_rad)])
    minor_direction = minor * np.linalg.norm(inverse @ [np.cos(angle_rad), -np.sin(angle_rad)])
    projected_extent = np.sqrt(8*np.log(2)*np.diag(covariance))
    return dict(beam_major_arcsec=major, beam_minor_arcsec=minor, beam_pa_deg=angle,
                beam_area_arcsec2=float(area), pixels_per_beam_major=float(fwhm[1]),
                pixels_per_beam_minor=float(fwhm[0]),
                pixels_per_beam_major_direction=float(major_direction),
                pixels_per_beam_minor_direction=float(minor_direction),
                pixels_per_beam_area=float(area/grid['pixel_area_arcsec2']),
                beams_width_x_projected=float(shape[1]/projected_extent[0]),
                beams_width_y_projected=float(shape[0]/projected_extent[1]),
                beams_width_over_bmaj=grid['fov_width_arcsec']/major,
                beams_width_over_bmin=grid['fov_width_arcsec']/minor,
                beams_area=float(grid['fov_area_arcsec2']/area),
                sampling_class=sampling_class(fwhm[0]),
                required_size_for_3pix=int(np.ceil(max(shape)*3/fwhm[0])))


def native_noise(image, wcs):
    """Existing peripheral sigma-clipped MAD, applied to untouched native data.

    Candidate = finite pixels outside central half-width × half-height rectangle
    (75% of rectangular area before NaNs). 8 iterations at most, symmetric 3σ,
    σ=1.4826022185 MAD about the median; >=32 retained pixels. No dilation.
    """
    mask = background_mask(image)
    values = image[mask].astype(float)
    median = float(np.median(values))
    sigma = float(1.4826022185*np.median(np.abs(values-median)))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Nonpositive/nonfinite native RMS')
    h, w = image.shape
    yy, xx = np.ogrid[:h, :w]
    candidate = np.isfinite(image) & ((xx < w/4) | (xx >= 3*w/4) | (yy < h/4) | (yy >= 3*h/4))
    initial = image[candidate]
    initial_sigma = float(1.4826022185*np.median(np.abs(initial-np.median(initial))))
    area = abs(np.linalg.det(local_jacobian(wcs, image.shape)*3600))
    return dict(rms_native=sigma, background_median=median,
                background_candidates=int(candidate.sum()), background_retained=int(mask.sum()),
                background_retained_fraction=float(mask.sum()/candidate.sum()),
                background_area_arcsec2_approx=float(mask.sum()*area),
                rms_before_clipping=initial_sigma, native_finite_fraction=float(np.isfinite(image).mean()))


def measure_source(path, target_wcs, shape, grid):
    row = dict(source_path=str(path), source_exists=Path(path).is_file(), beam_valid=False,
               noise_valid=False, unit_valid=False, sampling_class='unknown',
               beam_error='', noise_error='', unit_error='', source_error='')
    if not row['source_exists']:
        row['source_error'] = 'Source FITS missing'
        return row
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            data, header, native_wcs = load_celestial_fits(path)
        row['fits_warnings'] = '; '.join(dict.fromkeys(str(w.message) for w in caught))
        row.update(native_height=data.shape[0], native_width=data.shape[1], bunit=str(header.get('BUNIT', '')),
                   **{key.lower()+'_header': header.get(key) for key in ('BMAJ', 'BMIN', 'BPA')})
        try:
            scales = np.linalg.norm(local_jacobian(native_wcs, data.shape)*3600, axis=0)
            row.update(native_pixel_scale_x_arcsec=float(scales[0]), native_pixel_scale_y_arcsec=float(scales[1]))
            if target_wcs is None:
                raise ValueError('Target WCS unavailable; native noise is still measured')
            row.update(beam_metrics(header, target_wcs, shape, grid), beam_valid=True)
            native_cov = beam_covariance(header, native_wcs, data.shape)
            row['native_pixels_per_beam_minor'] = float(np.sqrt(np.linalg.eigvalsh(native_cov).min()*8*np.log(2)))
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            row['beam_error'] = str(exc)
        try:
            multiplier, unit = canonical_unit(row['bunit'])
            row.update(unit_multiplier=multiplier, canonical_unit=unit, unit_valid=True)
        except (ValueError, TypeError) as exc:
            row['unit_error'] = str(exc)
        try:
            row.update(native_noise(data, native_wcs), noise_valid=True)
            if row['unit_valid']:
                row['rms_canonical'] = row['rms_native'] * row['unit_multiplier']
            if row['beam_valid']:
                row['background_beams_approx'] = row['background_area_arcsec2_approx']/row['beam_area_arcsec2']
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            row['noise_error'] = str(exc)
    except Exception as exc:
        row['source_error'] = f'{type(exc).__name__}: {exc}'
    return row


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def pair_metrics(metrics, threshold=.8, difference_threshold=.1):
    pairs = []
    for cid, group in metrics.groupby('cluster_id', sort=False):
        by_name = {r['modality']: r for r in group.to_dict('records')}
        raw, sub = by_name['RAW'], by_name['T50kpcSUB']
        rc, tc = _number(raw.get('coverage')), _number(sub.get('coverage'))
        both_beams = bool(raw['beam_valid'] and sub['beam_valid'])
        both_noise = bool(raw['noise_valid'] and sub['noise_valid'])
        compatible = bool(raw['unit_valid'] and sub['unit_valid'] and
                          raw.get('canonical_unit') == sub.get('canonical_unit'))
        r = dict(cluster_id=cid, both_sources=bool(raw['source_exists'] and sub['source_exists']),
                 both_beams=both_beams, both_noise=both_noise, units_compatible=compatible,
                 usable_pair=bool(both_beams and both_noise and compatible),
                 coverage_raw=rc, coverage_t50kpcsub=tc,
                 coverage_difference=abs(rc-tc) if rc is not None and tc is not None else None,
                 raw_low_coverage=rc is not None and rc < threshold,
                 tapered_low_coverage=tc is not None and tc < threshold,
                 rms_ratio=None, beam_minor_ratio=None, beam_area_ratio=None,
                 raw_under2_tapered_ge2=False, tapered_under2_raw_ge2=False,
                 raw_class=raw['sampling_class'], tapered_class=sub['sampling_class'])
        r['large_coverage_difference'] = r['coverage_difference'] is not None and r['coverage_difference'] >= difference_threshold
        if both_noise and compatible:
            r['rms_ratio'] = sub['rms_canonical']/raw['rms_canonical']
            r['rms_ratio_unit'] = raw['canonical_unit']
        if both_beams:
            r['beam_minor_ratio'] = sub['beam_minor_arcsec']/raw['beam_minor_arcsec']
            r['beam_area_ratio'] = sub['beam_area_arcsec2']/raw['beam_area_arcsec2']
            r['raw_under2_tapered_ge2'] = raw['pixels_per_beam_minor'] < 2 <= sub['pixels_per_beam_minor']
            r['tapered_under2_raw_ge2'] = sub['pixels_per_beam_minor'] < 2 <= raw['pixels_per_beam_minor']
        pairs.append(r)
    return pd.DataFrame(pairs)


def _csv(path, frame):
    _atomic(path, lambda p: frame.to_csv(p, index=False))


def run_diagnostic(config, *, progress=True):
    """Measure one row per catalogue cluster/product, including missing sources."""
    validation, output = Path(config['validation_dir']).resolve(), Path(config['output_dir']).resolve()
    project = Path(__file__).resolve().parents[3]
    protected = [validation, project/'processed', project/'results', project/'scratch/data', project/'clusterprep']
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Diagnostic output must be independent of validation, sources, code and training caches')
    marker = output / '.radio_resolution_diagnostic'
    if output.exists() and any(output.iterdir()) and not marker.exists():
        raise ValueError('Output directory is not owned by this diagnostic')
    summary_path = validation/'summary.csv'
    summary = pd.read_csv(summary_path)
    if summary.cluster_id.duplicated().any():
        raise ValueError('Duplicate cluster IDs in summary')
    if config['cluster_ids'] is not None:
        unknown = set(config['cluster_ids']) - set(summary.cluster_id)
        if unknown:
            raise ValueError(f'Unknown IDs: {sorted(unknown)}')
        summary = summary[summary.cluster_id.isin(config['cluster_ids'])]
    if summary.empty:
        raise ValueError('Empty selection')
    output.mkdir(parents=True, exist_ok=True); marker.touch()
    from . import io, radio_noise
    import astropy
    measurement_code = '\n'.join(inspect.getsource(f) for f in
                                (sampling_class, canonical_unit, _sky_area, grid_metrics,
                                 beam_metrics, native_noise, measure_source, _clean, _number))
    implementation = dict(version=2, code={Path(p).name: fingerprint(p)['sha256'] for p in
                          (io.__file__, radio_noise.__file__)},
                          measurement_sha256=hashlib.sha256(measurement_code.encode()).hexdigest(),
                          numpy=np.__version__, astropy=astropy.__version__)
    manifest = dict(config=config, implementation=implementation, module=fingerprint(__file__), summary=fingerprint(summary_path),
                    validation_configuration=fingerprint(validation/'configuration.json'),
                    noise_method='native peripheral background_mask: up to 8 symmetric 3-sigma MAD iterations',
                    sampling_thresholds=[1, 2, 3], coverage_threshold=config['coverage_threshold'])
    _json(output/'configuration.json', manifest)
    rows, reused = [], 0
    for record in tqdm(summary.to_dict('records'), desc='Beams et bruit natif', disable=not progress):
        cid = record['cluster_id']
        if Path(cid).name != cid or cid in ('', '.', '..'):
            raise ValueError(f'Unsafe cluster ID: {cid}')
        meta_path = validation/'processed'/cid/'metadata.json'
        meta = {}
        try:
            meta = json.loads(meta_path.read_text())
            wcs = WCS(fits.Header(meta['wcs'])).celestial
            size = int(meta['identity']['settings']['config']['output_size'])
            shape = (size, size)
            grid = grid_metrics(wcs, shape)
            if not np.isclose(grid['pixel_scale_arcsec'], float(record['pixel_scale_arcsec']), rtol=.01):
                raise ValueError('Summary pixel scale inconsistent with target WCS')
            grid_error = ''
        except Exception as exc:
            wcs, shape, grid = None, None, {}
            grid_error = f'{type(exc).__name__}: {exc}'
        for product in PRODUCTS:
            source = meta.get('sources', {}).get(product)
            # Reject writes into input-source trees even with nonstandard paths.
            if source and (output == Path(source).resolve().parent or output in Path(source).resolve().parents):
                raise ValueError('Diagnostic output contains a source FITS')
            identity = dict(implementation=implementation, metadata=fingerprint(meta_path),
                            source=fingerprint(source) if source else None,
                            record=_clean(record), modality=product)
            checkpoint = output/'checkpoints'/cid/f'{product}.json'
            if checkpoint.exists() and not config['overwrite']:
                try:
                    cached = json.loads(checkpoint.read_text())
                    if cached['identity'] == identity:
                        rows.append(cached['row']); reused += 1
                        continue
                except (OSError, KeyError, ValueError):
                    pass
            row = dict(cluster_id=cid, modality=product, redshift=_number(record.get('redshift')),
                       fov_method=record.get('fov_method'), fov_arcmin=_number(record.get('fov_arcmin')),
                       coverage=_number(record.get('coverage_'+product.lower())), grid_error=grid_error,
                       validation_metadata=str(meta_path), **grid)
            original_source = meta.get('identity', {}).get('sources', {}).get(product)
            row['validation_source_matches'] = (identity['source'] == original_source
                                                 if original_source is not None else None)
            if source:
                row.update(measure_source(source, wcs, shape, grid))
            else:
                row.update(source_path=source, source_exists=bool(source and Path(source).is_file()),
                           beam_valid=False, noise_valid=False, unit_valid=False, sampling_class='unknown',
                           source_error='Target grid or source path unavailable', beam_error='', noise_error='', unit_error='')
            if row['beam_valid'] and row['coverage'] is not None:
                row['covered_beams_area_approx'] = row['beams_area']*row['coverage']
            if source and identity['source'] != fingerprint(source):
                raise RuntimeError('Source changed during diagnostic; rerun')
            row = _clean(row)
            _json(checkpoint, dict(identity=identity, row=row))
            rows.append(row)
    metrics = pd.DataFrame(rows)
    # Stable numerical columns, including a run with only missing/error inputs.
    for name in ('pixels_per_beam_minor', 'pixels_per_beam_major', 'pixel_scale_arcsec',
                 'beam_minor_arcsec', 'beam_major_arcsec', 'beam_area_arcsec2', 'beams_area',
                 'beams_width_over_bmaj', 'rms_canonical', 'rms_native', 'required_size_for_3pix',
                 'background_retained_fraction', 'native_pixels_per_beam_minor'):
        if name not in metrics:
            metrics[name] = np.nan
    if 'canonical_unit' not in metrics:
        metrics['canonical_unit'] = None
    metrics['low_coverage'] = metrics.coverage.notna() & (metrics.coverage < config['coverage_threshold'])
    pairs = pair_metrics(metrics, config['coverage_threshold'], config['coverage_difference_threshold'])
    _csv(output/'metrics.csv', metrics); _csv(output/'pairs.csv', pairs)
    _csv(output/'coverage_issues.csv', pairs[pairs.raw_low_coverage | pairs.tapered_low_coverage | pairs.large_coverage_difference])
    _csv(output/'sampling_discordance.csv', pairs[pairs.raw_under2_tapered_ge2 | pairs.tapered_under2_raw_ge2])
    issues = metrics[~metrics.source_exists.astype(bool) | ~metrics.beam_valid.astype(bool) |
                     ~metrics.noise_valid.astype(bool) | ~metrics.unit_valid.astype(bool) | metrics.grid_error.ne('') |
                     metrics.validation_source_matches.eq(False)]
    _csv(output/'metadata_issues.csv', issues)
    plot_diagnostics(metrics, pairs, output, config)
    gallery = plot_gallery(metrics, output, config)
    _json(output/'gallery_selection.json', gallery)
    write_report(metrics, pairs, output, config, reused, gallery)
    return metrics, pairs, dict(rows=len(metrics), reused=reused, clusters=len(summary))


def plot_diagnostics(metrics, pairs, output, config):
    import matplotlib.pyplot as plt
    folder = output/'figures'; folder.mkdir(exist_ok=True)
    def finish(fig, name):
        fig.tight_layout(); fig.savefig(folder/f'{name}.png', dpi=config['dpi']); plt.close(fig)
    def hist(ax, values, label, log=False):
        values = pd.to_numeric(values, errors='coerce').dropna().to_numpy()
        if log:
            values = values[values > 0]
        if len(values):
            low, high = values.min(), values.max()
            if low == high:
                low, high = (low*.9, high*1.1) if log else (low-.5, high+.5)
            edges = np.geomspace(low, high, 31) if log else np.linspace(low, high, 31)
            ax.hist(values, bins=edges, histtype='step', linewidth=1.8, label=f'{label} (n={len(values)})')
            if log:
                ax.set_xscale('log')
        else:
            ax.plot([], [], label=f'{label}: aucune mesure')
        ax.set_ylabel('Nombre d’images'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for product in PRODUCTS:
        part = metrics[metrics.modality == product]
        for ax, axis in zip(axes, ('minor', 'major')):
            hist(ax, part[f'pixels_per_beam_{axis}'], product, True)
            ax.set_xlabel(f'FWHM du beam en pixels — axe {axis}')
    for ax in axes:
        for limit in (1, 2, 3):
            ax.axvline(limit, color='gray', ls=':', lw=1)
    fig.suptitle('Beam natif représenté dans la grille cible ; seuils indicatifs 1, 2, 3')
    finish(fig, '01_pixels_per_beam')
    fig, ax = plt.subplots(figsize=(7, 4))
    hist(ax, metrics.drop_duplicates('cluster_id').pixel_scale_arcsec, 'Grille commune', True)
    ax.set_xlabel('Échelle équivalente √|det J| [arcsec/pixel]')
    finish(fig, '02_pixel_scales')
    fig, ax = plt.subplots(figsize=(8, 5))
    for product in PRODUCTS:
        part = metrics[(metrics.modality == product) & metrics.beam_valid.astype(bool)]
        ax.scatter(part.pixel_scale_arcsec, part.beam_minor_arcsec, s=12, alpha=.65, label=product)
    valid = metrics.pixel_scale_arcsec.dropna()
    xx = np.geomspace(max(valid.min(), .01), max(valid.max(), .02), 100) if len(valid) else np.array([1, 100])
    for n in (1, 2, 3):
        ax.plot(xx, n*xx, '--', label=f'{n} pixels/FWHM (pixels isotropes)')
    ax.set(xscale='log', yscale='log', xlabel='Échelle équivalente [arcsec/pixel]', ylabel='BMIN natif [arcsec]')
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    finish(fig, '03_scale_vs_bmin')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for product in PRODUCTS:
        part = metrics[metrics.modality == product]
        hist(axes[0], part.beams_width_over_bmaj, product, True)
        hist(axes[1], part.beams_area, product, True)
    axes[0].set_xlabel('Largeur céleste / BMAJ (compte linéaire)')
    axes[1].set_xlabel('Aire céleste / aire du beam (compte en aire)')
    finish(fig, '04_beam_counts')
    # Separate panels by unit: never pool Kelvin, Jy/sr and Jy/beam.
    units = sorted(metrics.loc[metrics.unit_valid.astype(bool), 'canonical_unit'].dropna().unique()) or ['aucune unité']
    fig, axes = plt.subplots(len(units), 2, figsize=(12, 4*len(units)), squeeze=False)
    for row, unit in enumerate(units):
        for product in PRODUCTS:
            part = metrics[(metrics.modality == product) & (metrics.canonical_unit == unit)]
            hist(axes[row, 0], part.rms_canonical, product, True)
        ratios = pairs.loc[pairs.get('rms_ratio_unit', pd.Series(index=pairs.index, dtype=object)) == unit, 'rms_ratio']
        hist(axes[row, 1], ratios, 'T50kpcSUB / RAW', True)
        axes[row, 0].set_xlabel(f'σ natif : MAD après clipping [{unit}]')
        axes[row, 1].set_xlabel('Rapport des σ natifs (beams distincts)')
        axes[row, 1].axvline(1, color='gray', ls=':')
    finish(fig, '05_native_noise')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for product in PRODUCTS:
        hist(axes[0], metrics.loc[metrics.modality == product, 'coverage'], product)
    axes[0].axvline(config['coverage_threshold'], color='gray', ls=':')
    axes[0].set_xlabel('Fraction couverte sur la grille commune')
    hist(axes[1], pairs.coverage_difference, '|RAW − T50kpcSUB|')
    axes[1].axvline(config['coverage_difference_threshold'], color='gray', ls=':')
    axes[1].set_xlabel('Différence absolue de couverture')
    finish(fig, '06_coverage')


def plot_gallery(metrics, output, config):
    import matplotlib.pyplot as plt
    selected = []
    valid = metrics[metrics.beam_valid.astype(bool)]
    for label, subset, ascending in [
        ('Confortable (≥3), sinon meilleur échantillonnage disponible', valid[valid.pixels_per_beam_minor >= 3], False),
        ('Sous-échantillonnage sévère', valid[valid.pixels_per_beam_minor < 1], True),
        ('Couverture insuffisante', metrics[metrics.low_coverage], True),
    ]:
        if subset.empty and label.startswith('Confortable'):
            subset = valid
        if subset.empty:
            selected.append(dict(category=label, cluster_id=None, reason='Aucun cas mesuré'))
            continue
        key = 'coverage' if label.startswith('Couverture') else 'pixels_per_beam_minor'
        chosen = subset.sort_values([key, 'cluster_id'], ascending=[ascending, True]).iloc[0]
        cid = chosen.cluster_id
        selected.append(dict(category=label, cluster_id=cid, modality=chosen.modality))
        fig = plt.figure(figsize=(11, 5), layout='constrained')
        try:
            for index, product in enumerate(PRODUCTS):
                row = metrics[(metrics.cluster_id == cid) & (metrics.modality == product)].iloc[0]
                source = Path(row.validation_metadata).parent/f'{product}.fits'
                if source.exists():
                    data, _, wcs = load_celestial_fits(source)
                    mask_path = source.with_name(product+'_MASK.fits')
                    mask = fits.getdata(mask_path).astype(bool) if mask_path.exists() else np.isfinite(data)
                    ax = fig.add_subplot(1, 2, index+1, projection=wcs)
                    image = normalize(data, mask=mask)
                    cmap = plt.get_cmap('viridis').with_extremes(bad='lightgray')
                    artist = ax.imshow(np.ma.masked_where(~mask, image), origin='lower', vmin=0, vmax=1,
                                       cmap=cmap, interpolation='nearest')
                    ax.set(xlabel='RA', ylabel='Dec', title=f'{product} | petit axe {row.pixels_per_beam_minor:.2f} pix\ncouverture {row.coverage:.1%}')
                    fig.colorbar(artist, ax=ax, shrink=.75, label='P30–P99 / asinh(10), par image')
                else:
                    ax = fig.add_subplot(1, 2, index+1)
                    ax.text(.5, .5, f'{product}\nImage de validation absente', ha='center'); ax.set_axis_off()
            caption = ('Échantillonnage confortable (≥3 pix)' if chosen.pixels_per_beam_minor >= 3
                       else 'Meilleur échantillonnage disponible') if label.startswith('Confortable') else label
            fig.suptitle(f'{cid} — sélection : {chosen.modality}\n{caption}\n'
                         'Grille commune existante · normalisation par image', fontsize=11)
            path = output/'figures'/f'gallery_{len(selected)}_{cid}.png'
            fig.savefig(path, dpi=config['dpi']); selected[-1]['figure'] = str(path.name)
        finally:
            plt.close(fig)
    return selected


def _stats(values):
    values = pd.to_numeric(values, errors='coerce').dropna()
    if not len(values):
        return '—'
    p = np.percentile(values, [5, 50, 95])
    return f'{p[1]:.4g} [{p[0]:.4g} ; {p[2]:.4g}]'


def write_report(metrics, pairs, output, config, reused, gallery):
    lines = ['# Diagnostic de résolution et de bruit RAW / T50kpcSUB', '',
             '## Périmètre et résultats', '',
             f"{metrics.cluster_id.nunique()} amas ; {len(metrics)} lignes amas/modalité. "
             f"{int(pairs.both_sources.sum())} paires avec les deux FITS, "
             f"{int(pairs.both_beams.sum())} avec deux beams valides, "
             f"{int(pairs.usable_pair.sum())} avec beams, RMS et unités comparables. "
             f"{reused} mesures reprises sans recalcul.", '',
             'Une paire « exploitable » ici ne constitue pas une sélection pour entraînement : '
             'la couverture et l’échantillonnage restent à examiner.', '',
             'Les tableaux donnent **médiane [P5 ; P95]**, en excluant seulement les valeurs '
             'manquantes de chaque statistique (effectifs indiqués séparément).', '',
             '| Quantité | RAW | T50kpcSUB |', '|---|---:|---:|']
    for title, key in [('BMAJ [arcsec]', 'beam_major_arcsec'), ('BMIN [arcsec]', 'beam_minor_arcsec'),
                       ('Aire beam [arcsec²]', 'beam_area_arcsec2'), ('FWHM petit axe [pixels cible]', 'pixels_per_beam_minor'),
                       ('FWHM petit axe [pixels natifs]', 'native_pixels_per_beam_minor'),
                       ('Largeur / BMAJ (linéaire)', 'beams_width_over_bmaj'), ('Aire FoV / aire beam', 'beams_area'),
                       ('Taille carrée requise pour ≥3 pix/FWHM au même FoV', 'required_size_for_3pix'),
                       ('Couverture', 'coverage'), ('Fraction du fond retenue après clipping', 'background_retained_fraction')]:
        lines.append(f"| {title} | " + ' | '.join(_stats(metrics.loc[metrics.modality == p, key]) for p in PRODUCTS) + ' |')
    unique = metrics.drop_duplicates('cluster_id')
    lines += ['', f"Échelle de pixels équivalente [arcsec/pixel], par amas : {_stats(unique.pixel_scale_arcsec)}. "
              f"Maximum : {unique.pixel_scale_arcsec.max():.4g} arcsec/pixel.", '',
              '## Échantillonnage', '', '| Produit | Beams valides | <1 pix | [1,2[ pix | [2,3[ pix | ≥3 pix | <2 pix |',
              '|---|---:|---:|---:|---:|---:|---:|']
    under = {}
    for product in PRODUCTS:
        part = metrics[(metrics.modality == product) & metrics.beam_valid.astype(bool)]
        total = len(part)
        entries = []
        for cls in CLASSES:
            number = int((part.sampling_class == cls).sum())
            entries.append(f'{number} ({number/total:.1%})' if total else '0 (—)')
        number = int((part.pixels_per_beam_minor < 2).sum()); under[product] = (number, total)
        lines.append(f'| {product} | {total} | ' + ' | '.join(entries) + f' | {number}/{total} |')
    discord = pairs[pairs.raw_under2_tapered_ge2 | pairs.tapered_under2_raw_ge2]
    lines += ['', 'Seuils indicatifs : <1 sévère, [1,2[ probable, [2,3[ limité, ≥3 plus confortable. '
              'Ni un théorème de Nyquist pour un beam gaussien, ni une garantie photométrique.', '',
              f"RAW <2 et T50kpcSUB ≥2 : **{int(pairs.raw_under2_tapered_ge2.sum())}** amas. "
              f"Inverse : **{int(pairs.tapered_under2_raw_ge2.sum())}**. "
              'Liste complète : [sampling_discordance.csv](sampling_discordance.csv).', '',
              'Premiers identifiants discordants : ' + (', '.join(discord.cluster_id.head(10)) if len(discord) else 'aucun')
              + (' ; liste exhaustive dans le CSV associé.' if len(discord) > 10 else '.'), '',
              f"Rapport BMIN(T50)/BMIN(RAW) : {_stats(pairs.beam_minor_ratio)}. "
              f"Rapport des aires de beam : {_stats(pairs.beam_area_ratio)}.", '',
              '## Bruit natif et unités', '', '| Unité canonique | RAW σ | T50kpcSUB σ | σ(T50)/σ(RAW) | Paires RMS |',
              '|---|---:|---:|---:|---:|']
    for unit in sorted(metrics.canonical_unit.dropna().unique()):
        ratio = pairs.loc[pairs.get('rms_ratio_unit', pd.Series(index=pairs.index, dtype=object)) == unit, 'rms_ratio']
        lines.append(f'| {unit} | ' + ' | '.join(_stats(metrics.loc[(metrics.modality == p) & (metrics.canonical_unit == unit), 'rms_canonical']) for p in PRODUCTS)
                     + f' | {_stats(ratio)} | {ratio.notna().sum()} |')
    lines += ['', 'Les RMS sont estimés **sur les images natives avant normalisation**, en unités du FITS '
              'puis convertis explicitement vers une unité canonique compatible. Aucun RMS des images '
              'interpolées n’est utilisé. Un même symbole Jy/beam ne désigne pas le même angle solide '
              'dans les deux produits ; les rapports décrivent ces produits à leurs résolutions natives.', '',
              'Algorithme réutilisé : `radio_noise.background_mask`. Candidats finis hors rectangle central '
              'de demi-largeur et demi-hauteur (75 % de la surface rectangulaire). À chaque itération : '
              'médiane, σ = 1,4826022185 × MAD, rejet symétrique au-delà de 3σ ; au plus 8 itérations '
              'ou stabilité du masque. Minimum 32 pixels conservés ; σ nul/non fini produit une erreur '
              'explicite. La valeur finale est une dispersion robuste autour de la médiane, pas '
              'sqrt(mean(I²)) incluant l’offset de fond. Aucun masquage par catalogue de sources ni '
              'dilatation des sources n’est effectué.', '',
              'Limites : ailes de sources, émission diffuse périphérique, résidus positifs/négatifs de '
              'soustraction et variations spatiales du bruit peuvent biaiser cette estimation. Les '
              'périphéries natives ne couvrent pas nécessairement les mêmes régions du ciel entre '
              'produits. Les pixels sont corrélés par le beam : le nombre de pixels retenus ne donne '
              'pas un nombre de mesures indépendantes. Les beams de fond indiqués sont des approximations '
              'géométriques, pas des degrés de liberté statistiques mesurés. Le clipping réduit légèrement '
              'la dispersion même pour un fond gaussien ; aucune correction ad hoc n’est appliquée.', '',
              '**Les écarts ne sont pas attribuables au tapering seul** : T50kpcSUB inclut aussi '
              'soustraction compacte, beam différent et potentiellement un champ natif différent.', '',
              '## Couverture et métadonnées', '',
              f"Seuil bas : {config['coverage_threshold']:.0%} ; différence importante : "
              f"≥{config['coverage_difference_threshold']:.0%} (points de fraction, pas différence relative).", '',
              f"RAW sous le seuil : {int(pairs.raw_low_coverage.sum())} ; T50kpcSUB : "
              f"{int(pairs.tapered_low_coverage.sum())} ; différence importante : {int(pairs.large_coverage_difference.sum())}.", '',
              '| Produit | Sources absentes | Beam invalide parmi sources présentes | RMS invalide parmi sources présentes | Unité invalide parmi sources présentes |',
              '|---|---:|---:|---:|---:|']
    for p in PRODUCTS:
        part = metrics[metrics.modality == p]; present = part.source_exists.astype(bool)
        lines.append(f'| {p} | {int((~present).sum())} | ' + ' | '.join(str(int((present & ~part[k].astype(bool)).sum())) for k in ('beam_valid','noise_valid','unit_valid')) + ' |')
    for title, flag in [('RAW sous le seuil', 'raw_low_coverage'), ('T50kpcSUB sous le seuil', 'tapered_low_coverage'), ('Écarts de couverture', 'large_coverage_difference')]:
        lines += ['', f"**{title} :** " + (', '.join(pairs.loc[pairs[flag], 'cluster_id']) or 'aucun') + '.']
    lines += ['', 'Détails : [coverage_issues.csv](coverage_issues.csv), [metadata_issues.csv](metadata_issues.csv). '
              'Les absences sont NaN dans les métriques de couverture, jamais assimilées à une couverture '
              'nulle mesurée. Aucune exclusion automatique.', '',
              f"Sources différentes de celles ayant produit la validation : "
              f"{int(metrics.validation_source_matches.eq(False).sum())}. "
              'Dans ce cas les couvertures historiques ne sont pas validées pour la nouvelle source ; '
              'mettre à jour séparément la validation avant une interprétation conjointe.', '',
              '## Géométrie et interprétation des beams', '',
              'La grille est lue dans les métadonnées de validation existantes. J est la Jacobienne '
              'locale pixels → est/nord, en arcsec/pixel. Les échelles x/y, les valeurs singulières et '
              'l’anisotropie sont enregistrées. Le beam gaussien céleste (BPA mesuré depuis le nord vers '
              'l’est) est transformé par J⁻¹ C J⁻ᵀ. Les racines des valeurs propres donnent les '
              'FWHM principales en pixels ; le plus petit axe classe l’échantillonnage. Pour des pixels '
              'isotropes, cela redonne exactement BMAJ/Δθ et BMIN/Δθ. Les longueurs des axes célestes '
              'projetés sont également sauvegardées pour les grilles anisotropes.', '',
              'Ωbeam = π BMAJ BMIN / (4 ln 2), en arcsec². L’aire du champ est la somme de polygones '
              'sphériques des pixels, pas un produit naïf de ΔRA et ΔDec. Les comptes **linéaires** '
              '(largeur/BMAJ, largeur/BMIN, et étendue projetée x/y) sont séparés du compte **en aire** '
              'ΩFoV/Ωbeam. Ce dernier est un nombre nominal d’éléments de résolution du beam natif, '
              'pas un nombre d’échantillons indépendants effectivement conservés à 128². '
              'La couverture multiplie ce compte pour une estimation du champ couvert. Le beam effectif '
              'après interpolation n’est ni ajusté ni mesuré ici.', '',
              'La Jacobienne du beam est évaluée au centre ; `area_vs_central_jacobian` mesure l’écart '
              'entre l’aire sphérique et l’approximation centrale. Les matrices CDELT, PC et CD sont '
              'interprétées par Astropy ; elles ne sont jamais réécrites pour ce diagnostic.', '',
              '## Comparaison des stratégies pour RR/RH', '',
              '| Stratégie | Avantages | Limites et conséquences |', '|---|---|---|',
              '| A — Grille commune 4R500, bilinéaire actuelle | Même région céleste, contexte périphérique et alignement multimodal. | Le pixel scale varie avec l’amas ; le beam natif peut être sous-échantillonné. Des structures compactes/filamenteuses et le bruit dépendent de la phase de rééchantillonnage. La géométrie commune ne suffit pas à une comparaison de résolution. |',
              '| B — Nombre de beams cible identique (`beam_crop`) | Contrôle le rapport FWHM/pixel si le nombre linéaire de beams est choisi avec soin. | Un même nombre de beams implique des champs différents quand les beams diffèrent ; peut supprimer les RR périphériques ou changer la fraction du halo incluse, ainsi que le contexte X-ray/optique. Fixer le nombre de beams seul ne garantit pas ≥3 pixels/FWHM. |',
              '| C — Grille commune avec anti-aliasing | Conserve le champ et l’alignement ; filtre les fréquences non représentables avant réduction. | Perte contrôlée de détails, corrélation et amplitude du bruit modifiées ; peut diluer les RR fins. Le lissage ne récupère pas l’information perdue et doit être accompagné d’une PSF effective et d’unités documentées. |', '',
              'Les anciens chemins `beam_crop` existent dans `dcreclass/data/loaders.py` et '
              '`dcreclass/data/processing.py` ; certaines variantes égalisent un champ moyen par version. '
              'La stratégie B ci-dessus est la définition demandée (nombre cible identique), sans '
              'supposer qu’une ancienne variante l’implémente exactement.', '',
              '### Recommandation fondée sur les mesures', '',
              f"À 128², {under['RAW'][0]}/{under['RAW'][1]} RAW et "
              f"{under['T50kpcSUB'][0]}/{under['T50kpcSUB'][1]} T50kpcSUB ont moins de 2 pixels "
              'sur le petit axe du beam. La distribution des tailles nécessaires pour 3 pixels/FWHM '
              'est donnée plus haut : elle quantifie le coût d’une conservation de la résolution native.', '',
              '**Privilégier C pour les premières comparaisons contrôlées à champ physique fixé**, '
              'en gardant exactement les mêmes entrées Chandra/Legacy et les mêmes amas/splits. '
              'Définir explicitement une résolution effective commune par paire, compatible avec la '
              'grille cible, puis un filtre anti-aliasing avant le rééchantillonnage. Une égalisation '
              'de PSF commune sert à isoler le produit/soustraction ; un anti-aliasing adapté séparément '
              'sans égalisation compare encore des résolutions différentes. Conserver A comme contrôle '
              'de géométrie, pas comme validation de la résolution native.', '',
              'Pour préserver les structures fines des RR, envisager une taille de grille supérieure '
              'ou une branche multi-échelle ; pour les RH diffus, tester le compromis sensibilité/échelle '
              'avec injections ou sources synthétiques. Utiliser B en ablation de sensibilité au beam, '
              'en signalant explicitement son changement de champ et sans lui attribuer un avantage '
              'de classification avant expérimentation. Le choix de la valeur cible du beam et du '
              'filtre reste à valider photométriquement ; **aucune de ces modifications n’est appliquée**.', '',
              '## Figures et reproductibilité', '']
    for name in ('01_pixels_per_beam','02_pixel_scales','03_scale_vs_bmin','04_beam_counts','05_native_noise','06_coverage'):
        lines.append(f'![{name}](figures/{name}.png)')
    for item in gallery:
        if item.get('figure'):
            lines += ['', f"![{item['category']}](figures/{item['figure']})"]
    lines += ['', 'Le [CSV principal](metrics.csv) contient une ligne par amas et produit ; '
              '[pairs.csv](pairs.csv) contient les rapports et indicateurs appariés. '
              '[configuration.json](configuration.json) conserve les seuils, versions, empreintes du '
              'code et du rapport de validation. Les checkpoints comparent les contenus des sources '
              'et des métadonnées pour éviter les recalculs. Les figures seules sont régénérées à la reprise.', '',
              'Modules réutilisés : `clusterprep.io.load_celestial_fits` (chargement), '
              '`radio_noise.background_mask`, `local_jacobian`, `beam_covariance` (mesures), '
              '`preprocessing.normalize` (galerie uniquement). '
              '`common_wcs_validation.project_plane` réalise la reprojection de validation existante ; '
              'ce diagnostic ne l’appelle pas et ne modifie aucun FITS.', '',
              'Références méthodologiques : '
              '[Astropy MAD](https://docs.astropy.org/en/stable/api/astropy.stats.mad_std.html) '
              '(conversion MAD en dispersion gaussienne) ; '
              '[Reproject](https://reproject.readthedocs.io/en/stable/celestial.html) '
              '(limites de l’interpolation lors des changements d’échelle et option adaptative avec anti-aliasing).', '']
    _atomic(output/'report.md', lambda p: p.write_text('\n'.join(lines)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = read_config(args.config, args.project_root)
    _, _, summary = run_diagnostic(config)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
