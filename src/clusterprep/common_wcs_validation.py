"""Standalone geometry validation. Never called by training or historical caches.

Bilinear reprojection preserves sampled native units, not integrated flux in
Jy/beam. No beam matching, photometric conversion, or compact-source removal.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import hashlib
import json
import os
import tempfile
import urllib.request

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from reproject import reproject_interp
from tqdm.auto import tqdm

from .io import load_celestial_fits, load_optical_fits
from .preprocessing import normalize

CATALOG_URL = 'https://lofar-surveys.org/public/clusters/table_full_sample.fits'
VERSION = 1
NORMALIZATION = dict(percentile_lo=30, percentile_hi=99, alpha=10.0)


@dataclass(frozen=True)
class ValidationConfig:
    output_size: int = 128
    fov_r500_factor: float = 4.0
    projection: str = 'TAN'
    legacy_bands: tuple = ('g', 'r', 'z')
    view: str = 'both'
    save_figures: bool = True
    save_processed: bool = True
    dpi: int = 180
    coverage_warning: float = 0.8

    def __post_init__(self):
        if not isinstance(self.output_size, int) or self.output_size < 1:
            raise ValueError('output_size must be a positive integer')
        if not np.isfinite(self.fov_r500_factor) or self.fov_r500_factor <= 0:
            raise ValueError('fov_r500_factor must be positive and finite')
        if self.projection not in ('TAN', 'SIN'):
            raise ValueError('projection must be TAN or SIN')
        if self.view not in ('before', 'after', 'both'):
            raise ValueError('view must be before, after or both')
        if not 0 <= self.coverage_warning <= 1 or self.dpi < 1:
            raise ValueError('Invalid coverage threshold or dpi')
        if not self.legacy_bands or len(set(self.legacy_bands)) != len(self.legacy_bands):
            raise ValueError('Select distinct Legacy bands')
        if any(b not in ('g', 'r', 'z') for b in self.legacy_bands):
            raise ValueError('Supported named Legacy bands: g, r, z')


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(fd)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _json(path, value):
    _atomic(path, lambda p: p.write_text(json.dumps(value, indent=2, allow_nan=False)))


def fingerprint(path):
    """Content identity, including missing files; detect changed inputs on resume."""
    path = Path(path).resolve()
    if not path.is_file():
        return {'path': str(path), 'missing': True}
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def fetch_catalog(path):
    """Download once; existing local snapshot is retained for reproducibility."""
    path = Path(path)
    if not path.exists():
        with urllib.request.urlopen(CATALOG_URL, timeout=60) as response:
            content = response.read()
        _atomic(path, lambda p: p.write_bytes(content))
    return path


def _finite(value, positive=False):
    if np.ma.is_masked(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and (not positive or number > 0) else None


def read_catalog(path):
    """Read unit-bearing FITS/ECSV; reject ambiguous unitless physical columns.

    Schema follows the official LoTSS-DR2/PSZ2 table. Coordinates are interpreted
    as equatorial ICRS/J2000 catalogue positions, never inferred from an image.
    """
    table = Table.read(path)
    required = ('Name', 'RA', 'DEC', 'z', 'R500', 'M500')
    if any(name not in table.colnames for name in required):
        raise ValueError(f'Catalogue requires columns {required}')
    columns = {}
    for name, unit in [('RA', u.deg), ('DEC', u.deg), ('R500', u.kpc), ('M500', u.Msun)]:
        if table[name].unit is None:
            raise ValueError(f'Catalogue column {name} lacks explicit units')
        columns[name] = np.ma.asarray(table[name]).filled(np.nan) * table[name].unit
        columns[name] = columns[name].to_value(unit)
    records = []
    seen = set()
    for i, row in enumerate(table):
        cid = str(row['Name']).replace(' ', '')
        if cid in seen or not cid or Path(cid).name != cid or cid in ('.', '..'):
            raise ValueError(f'Invalid or duplicate catalogue identifier: {cid}')
        seen.add(cid)
        records.append(dict(cluster_id=cid, ra_deg=_finite(columns['RA'][i]),
                            dec_deg=_finite(columns['DEC'][i]), redshift=_finite(row['z'], True),
                            r500_kpc=_finite(columns['R500'][i], True),
                            m500_msun=_finite(columns['M500'][i], True)))
    return records


def physical_field(record, factor=4.0, cosmology=Planck18):
    z = _finite(record.get('redshift'), True)
    radius = _finite(record.get('r500_kpc'), True)
    mass = _finite(record.get('m500_msun'), True)
    if z is None:
        return None
    method = 'r500'
    if radius is None and mass is not None:
        density = cosmology.critical_density(z).to_value(u.Msun / u.kpc**3)
        radius = (3 * mass / (4 * np.pi * 500 * density)) ** (1 / 3)
        method = 'm500_derived'
    if radius is None:
        return None
    angular = (factor * radius / cosmology.angular_diameter_distance(z).to_value(u.kpc)
               * u.rad).to_value(u.arcmin)
    if not np.isfinite(angular) or angular <= 0:
        return None
    return dict(r500_kpc=radius, fov_method=method, fov_arcmin=float(angular))


def median_field(records, config=ValidationConfig(), cosmology=Planck18):
    """Validation only: all complete catalogue rows, not training statistics."""
    fields = [physical_field(r, config.fov_r500_factor, cosmology) for r in records]
    values = [f['fov_arcmin'] for f in fields if f is not None]
    if not values:
        raise ValueError('No valid physical fields available for angular fallback')
    return float(np.median(values)), len(values)


def common_grid(record, fallback_arcmin, config=ValidationConfig(), cosmology=Planck18):
    ra, dec = _finite(record.get('ra_deg')), _finite(record.get('dec_deg'))
    if ra is None or dec is None or not 0 <= ra < 360 or not -90 <= dec <= 90:
        raise ValueError('Missing or invalid catalogue RA/Dec; no FITS-centre fallback')
    field = physical_field(record, config.fov_r500_factor, cosmology)
    if field is None:
        if _finite(fallback_arcmin, True) is None:
            raise ValueError('Invalid angular fallback')
        field = dict(r500_kpc=record.get('r500_kpc'), fov_method='angular_fallback',
                     fov_arcmin=float(fallback_arcmin))
    scale = field['fov_arcmin'] / 60 / config.output_size
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = [f'RA---{config.projection}', f'DEC--{config.projection}']
    wcs.wcs.cunit = ['deg', 'deg']
    wcs.wcs.radesys = 'ICRS'
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [(config.output_size + 1) / 2] * 2
    wcs.wcs.cdelt = [-scale, scale]
    wcs.array_shape = (config.output_size,) * 2
    return wcs, {**field, 'pixel_scale_arcsec': scale * 3600}


def discover_sources(record, fits_root):
    """Unclassified per-cluster downloads: independent of RAW availability."""
    cid = record['cluster_id']
    root = Path(fits_root) / cid
    return {name: str(root / f'{cid}{suffix}.fits') for name, suffix in
            [('RAW', ''), ('T50kpcSUB', 'T50kpcSUB'), ('CHANDRA', 'CHANDRA'), ('LEGACY', '_legacy')]}


def project_plane(image, source_wcs, target_wcs, size):
    image = np.where(np.isfinite(image), image, np.nan)
    projected, footprint = reproject_interp((image, source_wcs), target_wcs,
                                           shape_out=(size, size), order='bilinear')
    mask = (footprint > 0) & np.isfinite(projected)
    projected[~mask] = np.nan
    return projected.astype(np.float32), mask


def preprocess_common(sources, target_wcs, config=ValidationConfig()):
    """Read/project independently; one broken modality does not discard others."""
    images, errors, missing = {}, {}, []
    for name in ('RAW', 'T50kpcSUB', 'CHANDRA', 'LEGACY'):
        path = sources.get(name)
        if not path or not Path(path).is_file():
            missing.append(name)
            continue
        try:
            if name == 'LEGACY':
                planes, header, wcs = load_optical_fits(path)
                bands = [str(header.get(f'BAND{i}', i)).strip() for i in range(len(planes))]
                selected = []
                for band in config.legacy_bands:
                    if band not in bands:
                        missing.append(f'LEGACY_{band}')
                    else:
                        selected.append((f'LEGACY_{band}', planes[bands.index(band)]))
                if not selected:
                    raise ValueError(f'No requested named bands; FITS bands: {bands}')
            else:
                plane, header, wcs = load_celestial_fits(path)
                selected = [(name, plane)]
            for key, plane in selected:
                data, mask = project_plane(plane, wcs, target_wcs, config.output_size)
                images[key] = dict(data=data, mask=mask, wcs=target_wcs.deepcopy(),
                                   metadata={k: header[k] for k in ('BMAJ', 'BMIN', 'BPA', 'BUNIT') if k in header})
        except Exception as exc:
            errors[name] = f'{type(exc).__name__}: {exc}'
    return images, errors, missing


def save_figure(path, record, field, images, missing, errors, config):
    # Object API + Agg: no pyplot registration and no notebook display hook.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import AsinhNorm, Normalize
    import matplotlib.pyplot as plt
    names = ['RAW', 'T50kpcSUB', 'CHANDRA'] + [f'LEGACY_{b}' for b in config.legacy_bands]
    versions = ['before', 'after'] if config.view == 'both' else [config.view]
    fig = Figure(figsize=(6 * len(versions), 3.7 * len(names)), layout='constrained')
    FigureCanvasAgg(fig)
    try:
        for row, name in enumerate(names):
            item = images.get(name)
            for col, version in enumerate(versions):
                ax = fig.add_subplot(len(names), len(versions), row * len(versions) + col + 1,
                                     projection=item['wcs'] if item else None)
                ax.set_title(f'{name} — {"avant" if version == "before" else "après"} normalisation')
                if item is None:
                    ax.text(.5, .5, 'Erreur FITS' if name.split('_')[0] in errors else 'Modalité / bande absente',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                data, mask = item['data'], item['mask']
                if version == 'after':
                    data = normalize(data, mask=mask, **NORMALIZATION)
                    norm, unit = Normalize(0, 1), 'sans unité'
                else:
                    values = data[mask]
                    lo, hi = np.percentile(values, [1, 99]) if values.size else (0., 1.)
                    if hi <= lo:
                        hi = lo + max(abs(lo) * 1e-6, 1e-8)
                    norm = AsinhNorm(linear_width=(hi-lo)/10, vmin=lo, vmax=hi)
                    unit = item['metadata'].get('BUNIT', 'unité FITS non renseignée')
                cmap = plt.get_cmap('inferno' if name == 'CHANDRA' else 'viridis').copy()
                cmap.set_bad('lightgray')
                artist = ax.imshow(np.ma.masked_where(~mask, data), origin='lower',
                                   cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
                ax.set_xlabel('RA'); ax.set_ylabel('Dec')
                ax.coords.grid(alpha=.2)
                ticks = norm.inverse(np.linspace(0, 1, 5))
                fig.colorbar(artist, ax=ax, shrink=.75, label=unit, ticks=ticks, format='%.2g')
        fig.suptitle(f"{record['cluster_id']} | z={record.get('redshift')} | R500={field['r500_kpc']} kpc\n"
                     f"{field['fov_method']} | côté {field['fov_arcmin']:.3f} arcmin | "
                     f"{field['pixel_scale_arcsec']:.3f} arcsec/pixel\n"
                     'Avant : rendu asinh P1–P99 par panneau ; gris : hors couverture')
        _atomic(path, lambda p: fig.savefig(p, dpi=config.dpi))
    finally:
        plt.close(fig)
        fig.clear()


def _save_images(directory, images):
    outputs = []
    for name, item in images.items():
        header = item['wcs'].to_header()
        header.update(item['metadata'])
        header['NORMALIZ'] = False
        header['CWVER'] = VERSION
        for suffix, data in [('', item['data']), ('_MASK', item['mask'].astype(np.uint8))]:
            h = header.copy()
            if suffix:
                h['BUNIT'] = '1'
            path = directory / f'{name}{suffix}.fits'
            _atomic(path, lambda p, d=data, h=h: fits.PrimaryHDU(d, h).writeto(p, overwrite=True, checksum=True))
            outputs.append(path)
    return outputs


def _implementation_identity():
    from . import preprocessing, io
    import astropy, reproject, matplotlib
    return dict(version=VERSION, code=[fingerprint(p)['sha256'] for p in
                (__file__, preprocessing.__file__, io.__file__)],
                versions=dict(numpy=np.__version__, astropy=astropy.__version__,
                              reproject=reproject.__version__, matplotlib=matplotlib.__version__))


def run_validation(records, fits_root, output_dir, *, config=ValidationConfig(),
                   cosmology=Planck18, cluster_ids=None, overwrite=False,
                   catalog_provenance=None, progress=True):
    """Checkpoint per cluster; resume only identical inputs and intact outputs.

    A changed/incomplete prior result is rebuilt only inside this owned output
    tree. OVERWRITE forces rebuilding even identical completed clusters.
    """
    output = Path(output_dir).resolve()
    source_root = Path(fits_root).resolve()
    project = Path(__file__).resolve().parents[3]
    protected = [source_root, project / 'processed', project / 'results', project / 'scratch/data']
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Output must be independent of source data, training caches and results')
    marker = output / '.common_wcs_validation'
    if output.exists() and any(output.iterdir()) and not marker.exists():
        # A downloaded catalogue in the dedicated inputs directory is allowed.
        if {p.name for p in output.iterdir()} != {'inputs'}:
            raise ValueError('Refusing nonempty output directory not owned by this validator')
    output.mkdir(parents=True, exist_ok=True)
    marker.touch()
    fallback, complete = median_field(records, config, cosmology)
    settings = dict(config=asdict(config), cosmology=repr(cosmology),
                    fallback_arcmin=fallback, fallback_complete_count=complete,
                    fallback_scope='all_complete_catalogue_rows_validation_only',
                    normalization=NORMALIZATION, normalization_scope='image',
                    catalog_provenance=catalog_provenance, implementation=_implementation_identity(),
                    units=dict(r500='kpc', m500='Msun', centre='ICRS deg'),
                    reprojection='bilinear; native units; no integrated-flux conservation or beam matching')
    # JSON canonicalization also normalizes tuples before provenance comparisons.
    settings = json.loads(json.dumps(settings))
    selected = records
    if cluster_ids is not None:
        wanted = {cluster_ids} if isinstance(cluster_ids, str) else set(cluster_ids)
        unknown = wanted - {r['cluster_id'] for r in records}
        if unknown:
            raise ValueError(f'Unknown catalogue IDs: {sorted(unknown)}')
        selected = [r for r in records if r['cluster_id'] in wanted]
    _json(output / 'configuration.json', {**settings, 'selected_clusters': [r['cluster_id'] for r in selected]})
    rows = []
    for record in tqdm(selected, desc='Grille WCS commune', disable=not progress):
        cid = record['cluster_id']
        if Path(cid).name != cid or cid in ('', '.', '..'):
            raise ValueError('Unsafe cluster identifier')
        directory = output / 'processed' / cid
        metadata_path = directory / 'metadata.json'
        sources = discover_sources(record, source_root)
        row = dict(cluster_id=cid, redshift=record.get('redshift'), r500=record.get('r500_kpc'),
                   fov_method=None, fov_arcmin=None, pixel_scale_arcsec=None,
                   **{f'has_{n.lower()}': Path(p).is_file() for n, p in sources.items()},
                   **{f'coverage_{n}': None for n in ('raw', 't50kpcsub', 'chandra', 'legacy')},
                   status='error', error_message='', low_coverage='', resumed=False)
        try:
            identity = dict(settings=settings, record=record,
                            sources={n: fingerprint(p) for n, p in sources.items()})
            if not overwrite and metadata_path.exists():
                try:
                    previous = json.loads(metadata_path.read_text())
                    if (previous['identity'] == identity and previous['summary']['status'] == 'ok'
                            and all(fingerprint(p) == f for p, f in previous['outputs'].items())):
                        row = {**previous['summary'], 'resumed': True}
                        rows.append(row)
                        continue
                except (KeyError, ValueError, OSError):
                    pass
            wcs, field = common_grid(record, fallback, config, cosmology)
            row.update(r500=field['r500_kpc'], **{k: field[k] for k in
                       ('fov_method', 'fov_arcmin', 'pixel_scale_arcsec')})
            images, errors, missing = preprocess_common(sources, wcs, config)
            coverage = {k: float(v['mask'].mean()) for k, v in images.items()}
            for name in ('RAW', 'T50kpcSUB', 'CHANDRA'):
                row['coverage_' + name.lower()] = coverage.get(name)
            optical = [v for k, v in coverage.items() if k.startswith('LEGACY_')]
            row['coverage_legacy'] = float(np.mean(optical)) if optical else None
            row['low_coverage'] = '; '.join(k for k, v in coverage.items() if v < config.coverage_warning)
            row['status'] = ('partial' if errors else 'ok') if images else 'no_data'
            row['error_message'] = '; '.join(f'{k}: {v}' for k, v in errors.items())
            outputs = _save_images(directory, images) if config.save_processed else []
            if config.save_figures:
                figure = output / 'figures' / f'{cid}_common_wcs.png'
                save_figure(figure, record, field, images, missing, errors, config)
                outputs.append(figure)
            # Guard against a source changed during processing.
            if identity['sources'] != {n: fingerprint(p) for n, p in sources.items()}:
                raise RuntimeError('Source files changed during processing; rerun this cluster')
            # Remove obsolete products from an earlier configuration, only after success.
            expected = {p.resolve() for p in outputs}
            if directory.exists():
                for old in directory.glob('*.fits'):
                    if old.resolve() not in expected:
                        old.unlink()
            _json(metadata_path, dict(identity=identity, cluster_id=cid, centre=dict(ra_deg=record['ra_deg'],
                  dec_deg=record['dec_deg'], frame='ICRS'), redshift=record.get('redshift'),
                  m500_msun=record.get('m500_msun'), **field, wcs=dict(wcs.to_header()),
                  available_modalities=list(images), missing=missing, errors=errors, coverage=coverage,
                  normalization=NORMALIZATION, cosmology=repr(cosmology), sources=sources,
                  summary=row, outputs={str(p.resolve()): fingerprint(p) for p in outputs}))
        except Exception as exc:
            row.update(status='error', error_message=f'{type(exc).__name__}: {exc}')
        rows.append(row)
        # Flush after every cluster so a stopped run has a usable summary.
        _write_summary(output / 'summary.csv', rows)
    _write_summary(output / 'summary.csv', rows)
    return rows


def _write_summary(path, rows):
    def write(p):
        with p.open('w', newline='') as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    _atomic(path, write)
