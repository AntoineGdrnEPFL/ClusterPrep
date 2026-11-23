"""Opt-in paired 4R500 radio preprocessing, independent of ClusterDataset/cache."""
from dataclasses import dataclass

import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18
from reproject import reproject_interp

from .common_wcs_validation import ValidationConfig, common_grid, physical_field, project_plane
from .io import load_celestial_fits
from .radio_psf import read_radio, match_native, target_beam

VERSION = 2


@dataclass
class MatchedPair:
    raw: np.ndarray
    t50kcpsub: np.ndarray
    raw_mask: np.ndarray
    t50kcpsub_mask: np.ndarray
    chandra: np.ndarray
    chandra_mask: np.ndarray
    wcs: object
    metadata: dict

    @property
    def common_radio_mask(self):
        return self.raw_mask & self.t50kcpsub_mask

    def tensors(self):
        """Two [1,H,W] radio tensors, one shared X-ray tensor, explicit masks.

        Invalid entries are zero only at this interface and always accompanied
        by masks. No normalization or device placement; CPU float32 works on MPS.
        """
        import torch
        result = {}
        for name in ('raw', 't50kcpsub', 'chandra'):
            mask = getattr(self, name+'_mask')
            result[name] = torch.from_numpy(np.where(mask, getattr(self, name), 0).astype(np.float32)[None])
            result[name+'_mask'] = torch.from_numpy(mask.copy()[None])
        return result


def project_matched(data, valid, source_wcs, output_wcs, shape):
    """Strict bilinear support: every contributing neighbour must be valid.

    Linear interpolation preserves surface brightness, not exact aperture flux;
    its small extra PSF is measured in synthetic tests. No second Gaussian filter.
    """
    image, footprint = reproject_interp((np.where(valid, data, 0), source_wcs), output_wcs,
                                        shape_out=shape, order='bilinear')
    weight, _ = reproject_interp((valid.astype(float), source_wcs), output_wcs,
                                 shape_out=shape, order='bilinear')
    mask = (footprint > 0) & np.isfinite(image) & (weight >= 1-1e-8)
    image[~mask] = np.nan
    return image.astype(np.float32), mask


def match_pair(raw, t50kcpsub, output_wcs, chandra, chandra_mask, *, size=128,
               pixels_per_beam=3., min_support=.999, output_unit='Jy/beam', truncate=5.):
    """Match two native RadioPlanes and attach an already preprocessed X-ray.

    The caller supplies CHANDRA on output_wcs, once, with its coverage. It is
    copied unchanged and never radio-smoothed. Output radio units are common.
    """
    if not isinstance(size, int) or size < 2:
        raise ValueError('size must be an integer >= 2')
    shape = (size, size)
    if np.shape(chandra) != shape or np.shape(chandra_mask) != shape:
        raise ValueError('CHANDRA and its mask must be on the output grid')
    if np.any(np.asarray(chandra_mask, bool) & ~np.isfinite(chandra)):
        raise ValueError('CHANDRA mask marks nonfinite observations as valid')
    if output_unit not in ('Jy/beam', 'Jy/sr'):
        raise ValueError('output_unit must be Jy/beam or Jy/sr')
    target, minimum = target_beam((raw, t50kcpsub), output_wcs, shape, pixels_per_beam)
    images, masks, diagnostics = [], [], {}
    for name, plane in (('RAW', raw), ('T50kpcSUB', t50kcpsub)):
        smoothed, valid, _, diagnostic = match_native(plane, target, min_support=min_support,
                                                     truncate=truncate)
        image, mask = project_matched(smoothed, valid, plane.wcs, output_wcs, shape)
        if output_unit == 'Jy/beam':
            image *= target.solid_angle_sr
        images.append(image)
        masks.append(mask)
        diagnostics[name] = {**diagnostic, 'output_coverage': float(mask.mean())}
    metadata = dict(version=VERSION, target_beam=vars(target), bunit=output_unit,
                    output_wcs=dict(output_wcs.to_header()), size=size,
                    target_pixels_per_beam=pixels_per_beam, minimum_pixels_per_beam=minimum,
                    nyquist_amplitude=float(np.exp(-.5*(np.pi*minimum/np.sqrt(8*np.log(2)))**2)),
                    reprojection='bilinear surface brightness; strict valid neighbours',
                    psf='nominal circular Gaussian; discrete/interpolation error requires validation',
                    products=diagnostics, chandra_coverage=float(np.mean(chandra_mask)))
    return MatchedPair(images[0], images[1], masks[0], masks[1],
                       np.array(chandra, copy=True), np.array(chandra_mask, dtype=bool, copy=True),
                       output_wcs.deepcopy(), metadata)


def prepare_matched_pair(raw_path, t50kcpsub_path, chandra_path, record, *, size=128,
                         pixels_per_beam=3., min_support=.999, output_unit='Jy/beam',
                         raw_mask=None, t50kcpsub_mask=None, truncate=5.):
    """Native FITS -> one 4R500 grid -> paired radio and a single CHANDRA input.

    No catalogue median fallback, normalization, labels, training, or cache IO.
    Missing physical metadata is an explicit error. The full requested field is
    retained even when observations cover only part of it.
    """
    if physical_field(record) is None:
        raise ValueError('4R500 requires valid redshift and R500 or M500')
    wcs, field = common_grid(record, None, ValidationConfig(output_size=size))
    raw = read_radio(raw_path, raw_mask)
    tapered = read_radio(t50kcpsub_path, t50kcpsub_mask)
    if chandra_path is None:
        xray = np.zeros((size, size), dtype=np.float32)
        xray_mask = np.zeros((size, size), dtype=bool)
        header = {}
    else:
        xray, header, xray_wcs = load_celestial_fits(chandra_path)
        xray, xray_mask = project_plane(xray, xray_wcs, wcs, size)
    result = match_pair(raw, tapered, wcs, xray, xray_mask, size=size,
                        pixels_per_beam=pixels_per_beam, min_support=min_support,
                        output_unit=output_unit, truncate=truncate)
    angular = float(result.metadata['target_beam']['major_arcsec'])
    physical = angular * u.arcsec.to(u.rad) * Planck18.angular_diameter_distance(
        record['redshift']).to_value(u.kpc)
    result.metadata.update(cluster_id=record.get('cluster_id'), field=field,
                           target_resolution=dict(fwhm_arcsec=angular, fwhm_kpc=float(physical),
                               fwhm_over_r500=float(physical / field['r500_kpc']),
                               r500_kpc=float(field['r500_kpc']), cosmology=str(Planck18)),
                           coverage=dict(RAW=float(result.raw_mask.mean()),
                               T50kpcSUB=float(result.t50kcpsub_mask.mean()),
                               CHANDRA=float(result.chandra_mask.mean()),
                               intersection=float(result.common_radio_mask.mean())),
                           has_chandra=chandra_path is not None,
                           chandra_status=('absent' if chandra_path is None else
                               'present_covered' if xray_mask.any() else 'present_uncovered'),
                           chandra_source=str(chandra_path) if chandra_path is not None else None,
                           chandra_bunit=header.get('BUNIT'),
                           chandra_preprocessing='common-grid bilinear; no radio PSF convolution')
    return result
