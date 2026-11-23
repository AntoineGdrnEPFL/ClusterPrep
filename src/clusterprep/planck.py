"""HEALPix Compton-y projection, reusable physical caches, and radio geometry.

Reprojection never changes the instrumental beam. No intensity scaling or
augmentation is stored. One source map is retained per process (DataLoader worker).
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_from_healpix

VERSION = 1


def validate_planck(options, modalities):
    options = dict(options or {})
    if set(options) - {'source_path', 'cache_dir', 'field', 'force_rebuild'}:
        raise ValueError('Unknown planck option; normalization belongs in normalization.planck')
    if 'planck' in (modalities or []):
        if 'radio' not in modalities:
            raise ValueError('Planck requires radio')
        if not options.get('source_path'):
            raise ValueError('planck.source_path is required')
    elif options:
        raise ValueError('planck options require explicit planck modality')
    return options


def source_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns)


@lru_cache(maxsize=1)
def _read_map(identity_json, field):
    identity = json.loads(identity_json)
    with fits.open(identity['path'], memmap=True) as hdus:
        tables = [h for h in hdus if h.header.get('PIXTYPE', '').strip() == 'HEALPIX']
        if len(tables) != 1:
            raise ValueError('Expected one HEALPix HDU')
        hdu = tables[0]
        header = hdu.header
        nside = int(header['NSIDE'])
        ordering = header['ORDERING'].strip().upper()
        frames = {'G': 'galactic', 'GALACTIC': 'galactic', 'C': 'icrs', 'EQUATORIAL': 'icrs'}
        coords = header['COORDSYS'].strip().upper()
        if nside < 1 or nside & (nside - 1) or ordering not in {'RING', 'NESTED'} or coords not in frames:
            raise ValueError('Unsupported NSIDE, ORDERING or COORDSYS')
        if header.get('INDXSCHM', 'IMPLICIT').strip() != 'IMPLICIT' or header.get('FIRSTPIX', 0) != 0:
            raise ValueError('Only full implicit HEALPix maps supported')
        if field not in hdu.columns.names:
            raise ValueError(f'HEALPix field {field!r} absent: {hdu.columns.names}')
        values = np.array(hdu.data[field], dtype=np.float64).reshape(-1)
        if values.size != 12 * nside * nside:
            raise ValueError('HEALPix size inconsistent with NSIDE')
        # HEALPix UNSEEN sentinel, FITS BAD_DATA, and non-finite samples.
        invalid = ~np.isfinite(values) | np.isclose(values, -1.6375e30, rtol=1e-5, atol=0)
        if 'BAD_DATA' in header:
            invalid |= values == header['BAD_DATA']
        values[invalid] = np.nan
        metadata = dict(nside=nside, ordering=ordering, frame=frames[coords],
                        unit=hdu.columns[field].unit or 'y', beam_arcmin=header.get('BEAMSIZE'))
    values.setflags(write=False)
    return values, metadata


def project(options, wcs, shape, reference='radio'):
    """Return unnormalized image, validity and cache provenance on any radio grid."""
    identity = source_identity(options['source_path'])
    field = options.get('field', 'FULL')
    header = wcs.celestial.to_header(relax=True)
    signature = dict(version=VERSION, source=identity, field=field, shape=list(shape),
                     wcs=header.tostring(), order='bilinear', reference=str(reference))
    payload = json.dumps(signature, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    root = Path(options.get('cache_dir', 'processed/planck_PR4'))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{digest}.fits'
    existed = path.exists()
    if existed and not options.get('force_rebuild', False):
        try:
            with fits.open(path, checksum=True) as hdus:
                if hdus[0].data is None:
                    raise ValueError('Empty Planck cache')
                image = hdus[0].data.astype(np.float32)
                mask = hdus['VALID'].data.astype(bool)
                if (hdus[0].header['CACHEKEY'] != digest or image.shape != tuple(shape)
                        or mask.shape != image.shape or not np.array_equal(mask, np.isfinite(image))
                        or WCS(hdus[0].header).celestial.to_header(relax=True).tostring() != header.tostring()
                        or any(h.verify_checksum() != 1 or h.verify_datasum() != 1 for h in hdus)):
                    raise ValueError('Invalid Planck cache')
            return dict(planck=image, planck_mask=mask, wcs=wcs,
                        cache_status='reused', cache_path=str(path))
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            pass
    try:
        values, metadata = _read_map(json.dumps(identity, sort_keys=True), field)
        image, footprint = reproject_from_healpix(
            (values, metadata['frame']), wcs, shape_out=tuple(shape),
            nested=metadata['ordering'] == 'NESTED', order='bilinear')
        mask = np.isfinite(image) & (footprint > 0)
        image = np.where(mask, image, np.nan).astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f'Planck projection failed for {reference}: {exc}') from exc
    if source_identity(options['source_path']) != identity:
        raise RuntimeError('Planck source changed during projection; retry with a stable source')
    header['BUNIT'] = metadata['unit']
    header['CACHEKEY'] = digest
    header['PLKNSIDE'] = metadata['nside']
    if metadata['beam_arcmin'] is not None:
        header['BMAJ'] = header['BMIN'] = metadata['beam_arcmin'] / 60
    hdus = fits.HDUList([fits.PrimaryHDU(image, header),
                         fits.ImageHDU(mask.astype('uint8'), name='VALID')])
    fd, temporary = tempfile.mkstemp(dir=root, suffix='.fits')
    os.close(fd)
    try:
        hdus.writeto(temporary, overwrite=True, checksum=True)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return dict(planck=image, planck_mask=mask, wcs=wcs,
                cache_status='rebuilt' if existed else 'created', cache_path=str(path))


def _read_reduced(path, digest, wcs, shape):
    with fits.open(path, checksum=True) as hdus:
        image = np.array(hdus[0].data, dtype=np.float32)
        mask = np.array(hdus['VALID'].data, dtype=bool)
        if (hdus[0].header['CACHEKEY'] != digest or image.shape != tuple(shape)
                or mask.shape != image.shape or not np.array_equal(mask, np.isfinite(image))
                or WCS(hdus[0].header).celestial.to_header(relax=True).tostring() != wcs.to_header(relax=True).tostring()
                or any(h.verify_checksum() != 1 or h.verify_datasum() != 1 for h in hdus)):
            raise ValueError('Invalid reduced Planck cache')
    return image, mask


def aligned_images(info, images, size, options, matched=False):
    """Physical, unaugmented reduced cache; exact historical block averages."""
    from .io import load_celestial_geometry, resolve_radio_path
    from .preprocessing import centered_raw_window, block_average
    path = resolve_radio_path(info)
    reference = str(Path(path).resolve())
    if matched:
        # No full-radio projection: it was unused on this path.
        return project(options, images['wcs'], images['raw'].shape, reference + ':matched')
    signature = dict(version=VERSION, reduction_version=1,
                     radio=source_identity(path), source=source_identity(options['source_path']),
                     field=options.get('field', 'FULL'), factor=images['factor'], size=size,
                     wcs=images['wcs'].to_header(relax=True).tostring())
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    root = Path(options.get('cache_dir', 'processed/planck_PR4')) / 'reduced'
    target = root / f'{digest}.fits'
    existed = target.exists()
    if existed and not options.get('force_rebuild', False):
        try:
            image, mask = _read_reduced(target, digest, images['wcs'], (size, size))
            return dict(planck=image, planck_mask=mask, cache_path=str(target), cache_status='reused')
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            pass
    shape, _, wcs = load_celestial_geometry(path)
    full = project(options, wcs, shape, reference)
    window = centered_raw_window(shape, size, images['factor'])
    reduced = block_average(full['planck'][window], images['factor'])
    mask = np.isfinite(reduced)
    if (source_identity(path) != signature['radio'] or
            source_identity(options['source_path']) != signature['source']):
        raise RuntimeError('Source changed during Planck reduction')
    header = images['wcs'].to_header(relax=True)
    # Read only the cached header; never reload the global HEALPix map for units.
    source_header = fits.getheader(full['cache_path'])
    for name in ('BUNIT', 'BMAJ', 'BMIN', 'PLKNSIDE'):
        if name in source_header:
            header[name] = source_header[name]
    header['CACHEKEY'] = digest
    header['NORMALIZ'] = False
    header['AUGMENT'] = False
    root.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=root, suffix='.fits')
    os.close(fd)
    try:
        fits.HDUList([fits.PrimaryHDU(reduced, header),
                      fits.ImageHDU(mask.astype('uint8'), name='VALID')]).writeto(
                          temporary, overwrite=True, checksum=True)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return dict(planck=reduced, planck_mask=mask, cache_path=str(target),
                cache_status='rebuilt' if existed else 'created')


def normalization_options(normalization, modalities):
    result = dict(normalization or {})
    if 'planck' in (modalities or []):
        options = {'method': 'standard', 'scope': 'train', **result.get('planck', {})}
        if options['method'] not in {'none', 'standard', 'percentile_asinh'}:
            raise ValueError('Planck normalization: none, standard or percentile_asinh')
        if options['scope'] != 'train':
            raise ValueError('Planck normalization must use train statistics')
        result['planck'] = options
    return result
