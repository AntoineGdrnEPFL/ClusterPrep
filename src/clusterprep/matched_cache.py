"""Atomic paired cache, isolated from historical processed/scattering files."""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from urllib.parse import quote
import zipfile

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from . import matched_resolution, radio_psf, common_wcs_validation, io
from .matched_resolution import MatchedPair, prepare_matched_pair
from .common_wcs_validation import common_grid, ValidationConfig
from .radio_psf import Beam, target_beam

CACHE_VERSION = 1
ARRAY_NAMES = ('raw', 't50kcpsub', 'chandra', 'raw_mask', 't50kcpsub_mask', 'chandra_mask')


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def _stat(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class MatchedPairStore:
    """One file per pair/physical configuration, shared by RAW and T50 datasets.

    Source SHA256 is recomputed when size/mtime/ctime/inode changes; memoized only
    in this process. A read after process restart rehashes all sources. Pair arrays
    are retained in RAM only with memory=True. Source files must remain available.
    """
    def __init__(self, root=None, *, memory=False):
        self.root = None if root is None else Path(root) / 'matched_resolution' / f'v{CACHE_VERSION}'
        self.memory = memory
        self._fingerprints = {}
        self._specifications = {}
        self._pairs = {}
        self.implementation = dict(
            cache_version=CACHE_VERSION, pipeline_version=matched_resolution.VERSION,
            code={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                  for m in (matched_resolution, radio_psf, common_wcs_validation, io)},
            libraries={n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'astropy', 'reproject')})

    def _fingerprint(self, path):
        if path is None:
            return None
        path = Path(path).resolve()
        signature = _stat(path)
        previous = self._fingerprints.get(str(path))
        if previous is None or previous[0] != signature:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            if signature != _stat(path):
                raise RuntimeError('Source changed while hashing')
            value = dict(path=str(path), sha256=digest.hexdigest(), bytes=signature[2])
            self._fingerprints[str(path)] = (signature, value)
        return deepcopy(self._fingerprints[str(path)][1])

    def specification(self, sources, record, size, parameters):
        """Resolve both native beams before the cache key, without convolution."""
        identity = dict(implementation=deepcopy(self.implementation), cluster_id=record['cluster_id'],
                        sources={n: self._fingerprint(p) for n, p in sources.items()},
                        record=record, size=size, preprocessing=parameters,
                        products=['RAW', 'T50kpcSUB'])
        token = canonical_json(identity)
        cid = record['cluster_id']
        previous = self._specifications.get(cid)
        if previous is not None and previous[0] == token:
            return previous[1], previous[2]
        wcs, _ = common_grid(record, None, ValidationConfig(output_size=size))
        beams = []
        for name in ('RAW', 'T50kpcSUB'):
            with fits.open(sources[name], memmap=True) as hdul:
                hdu, _ = io._find_celestial_image_hdu(hdul)
                beams.append(SimpleNamespace(beam=Beam.from_header(hdu.header)))
        beam, _ = target_beam(beams, wcs, (size, size), parameters['pixels_per_beam'])
        identity.update(target_beam=vars(beam), output_wcs=dict(wcs.to_header()))
        digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
        self._specifications[cid] = (token, deepcopy(identity), digest)
        return identity, digest

    def path(self, identity, digest):
        if self.root is None:
            return None
        cid = quote(str(identity['cluster_id']), safe='+')
        if cid in ('', '.', '..'):
            raise ValueError('Invalid cluster_id')
        return self.root / cid / f'{digest}.npz'

    @staticmethod
    def _read(path, identity):
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(str(archive['manifest'].item()))
            if manifest['identity'] != identity:
                raise ValueError('Invalid paired cache provenance')
            arrays = {n: archive[n].copy() for n in ARRAY_NAMES}
        for name, array in arrays.items():
            if array.shape != (identity['size'],)*2 or array_digest(array) != manifest['digests'][name]:
                raise ValueError('Invalid paired cache array')
            if name.endswith('_mask') and array.dtype != np.bool_:
                raise ValueError('Invalid mask dtype')
        for name in ('raw', 't50kcpsub', 'chandra'):
            if name == 'chandra' and identity['sources']['CHANDRA'] is None:
                if arrays[name+'_mask'].any() or not np.all(arrays[name] == 0):
                    raise ValueError('Absent CHANDRA must have zero image and mask')
                continue
            if not np.array_equal(np.isfinite(arrays[name]), arrays[name+'_mask']):
                raise ValueError('Invalid paired cache coverage')
        metadata = manifest['metadata']
        if metadata['target_beam'] != identity['target_beam']:
            raise ValueError('Invalid target beam')
        wcs = WCS(identity['output_wcs'])
        wcs.array_shape = (identity['size'],)*2
        return MatchedPair(**arrays, wcs=wcs, metadata=metadata)

    @staticmethod
    def _write(path, pair, identity):
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {n: getattr(pair, n) for n in ARRAY_NAMES}
        manifest = dict(identity=identity, metadata=pair.metadata,
                        digests={n: array_digest(a) for n, a in arrays.items()})
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.building-', suffix='.npz')
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez_compressed(stream, manifest=canonical_json(manifest), **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def get(self, sources, record, size, parameters):
        identity, digest = self.specification(sources, record, size, parameters)
        cid = record['cluster_id']
        path = self.path(identity, digest)
        previous = self._pairs.get(cid)
        if self.memory and previous is not None and previous[0] == digest:
            pair, status = previous[1], 'memory'
        else:
            pair = None
            existed = path is not None and path.exists()
            if existed:
                try:
                    pair = self._read(path, identity)
                except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                    pass
            status = 'reused'
            if pair is None:
                pair = prepare_matched_pair(sources['RAW'], sources['T50kpcSUB'], sources['CHANDRA'],
                                             record, size=size, **parameters)
                if pair.metadata['target_beam'] != identity['target_beam']:
                    raise RuntimeError('Target beam changed during preprocessing')
                # No publication of data computed from changing inputs.
                if identity != self.specification(sources, record, size, parameters)[0]:
                    raise RuntimeError('Sources changed during matched preprocessing')
                pair.metadata.update(cache_key=digest, cache_identity=identity,
                                     chandra_sha256=array_digest(pair.chandra),
                                     chandra_mask_sha256=array_digest(pair.chandra_mask))
                if path is not None:
                    self._write(path, pair, identity)
                status = ('rebuilt' if existed else 'created') if path is not None else 'uncached'
            if self.memory:
                self._pairs[cid] = (digest, pair)
        # Verify even on disk/RAM hits; concurrent source edits fail explicitly.
        if identity != self.specification(sources, record, size, parameters)[0]:
            raise RuntimeError('Sources changed while loading matched pair')
        return pair, dict(key=digest, path=str(path) if path is not None else None, status=status)
