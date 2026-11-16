"""Dataset adapter for opt-in matching; never edits keys, labels or split state."""
from copy import deepcopy
from dataclasses import dataclass, field
from collections.abc import Mapping

import numpy as np
import torch

from .augmentation import decode_variant, transform_aligned
from .io import resolve_radio_path
from .labels import SUBTYPE_TASKS, labels_for, task_target
from .matched_cache import MatchedPairStore
from .modalities import normalize_planes


@dataclass(frozen=True)
class MatchedResolutionConfig:
    radio_mask: str = 'intersection'
    target_pixels_per_beam: float = 3.
    min_support: float = .999
    truncate: float = 5.
    output_unit: str = 'Jy/beam'
    coverage_thresholds: dict = field(default_factory=dict)
    coverage_policy: str = 'error'

    def __post_init__(self):
        if self.radio_mask not in ('intersection', 'specific'):
            raise ValueError('radio_mask must be intersection or specific')
        if self.coverage_policy not in ('error', 'report'):
            raise ValueError('coverage_policy must be error or report; no silent exclusions')
        for name, value, low, high in (
            ('target_pixels_per_beam', self.target_pixels_per_beam, 0., np.inf),
            ('min_support', self.min_support, 0., 1.),
            ('truncate', self.truncate, 0., np.inf),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low < value <= high:
                raise ValueError(f'Invalid {name}')
        if self.truncate < 4:
            raise ValueError('truncate must be >= 4 sigma')
        if self.output_unit not in ('Jy/beam', 'Jy/sr'):
            raise ValueError('output_unit must be Jy/beam or Jy/sr')
        if not isinstance(self.coverage_thresholds, Mapping):
            raise ValueError('coverage_thresholds must be a mapping')
        thresholds = dict(self.coverage_thresholds)
        if set(thresholds) - {'RAW', 'T50kpcSUB', 'CHANDRA', 'intersection'}:
            raise ValueError('Unknown coverage threshold')
        for name, value in thresholds.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid coverage threshold for {name}')
        object.__setattr__(self, 'coverage_thresholds', thresholds)

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return deepcopy(value)
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError('matched_resolution must be a mapping or MatchedResolutionConfig')
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError('Unknown matched_resolution option') from exc

    def preprocessing_parameters(self):
        # Mask/eligibility are read-time policies, not costly convolution inputs.
        return dict(pixels_per_beam=self.target_pixels_per_beam, min_support=self.min_support,
                    truncate=self.truncate, output_unit=self.output_unit)


class CoverageEligibilityError(ValueError):
    def __init__(self, metadata):
        self.metadata = deepcopy(metadata)
        super().__init__(f"{metadata['cluster_id']}: insufficient coverage: " +
                         '; '.join(metadata['eligibility']['reasons']))


def sample_metadata(metadata):
    """Stable collatable subset; full metadata lives in get_resolution_metadata."""
    resolution = metadata['target_resolution']
    return dict(coverage=deepcopy(metadata['coverage']),
                has_chandra=metadata['has_chandra'], chandra_status=metadata['chandra_status'],
                target_resolution={n: resolution[n] for n in
                                   ('fwhm_arcsec', 'fwhm_kpc', 'fwhm_over_r500', 'r500_kpc')},
                eligible=metadata['eligibility']['eligible'], radio_mask=metadata['radio_mask'],
                radio_source=metadata['radio_source'], cache_key=metadata['cache_key'],
                selected_radio_coverage=metadata['selected_radio_coverage'])


class MatchedDatasetAdapter:
    def __init__(self, dataset, options, records, cache_root):
        self.options = MatchedResolutionConfig.parse(options)
        self.records = deepcopy(records or {})
        self.store = MatchedPairStore(cache_root, memory=dataset.cache)
        if dataset.size < 2:
            raise ValueError('matched_resolution requires size >= 2')
        if dataset.scattering_dir is not None:
            raise ValueError('matched_resolution cannot reuse historical precomputed scattering')
        if dataset.radio_noise['enabled']:
            raise ValueError('matched_resolution radio_noise requires a matched-beam noise model')
        if dataset.modalities is not None and set(dataset.modalities) - {'radio', 'xray'}:
            raise ValueError('matched_resolution currently supports radio/xray modalities only')
        if dataset.modalities is not None and not dataset.include_masks:
            raise ValueError('matched_resolution explicit modalities require include_masks=True')
        # Validate required identities without processing images or filtering keys.
        for key in dataset.keys:
            info = dataset.data_info[key]
            self.sources(info)
            self.record(info)

    @staticmethod
    def sources(info):
        if info.get('radio_source', 'RAW') not in ('RAW', 'T50kpcSUB'):
            raise ValueError('matched_resolution radio_source must be RAW or T50kpcSUB')
        return dict(RAW=resolve_radio_path(info, 'RAW'),
                    T50kpcSUB=resolve_radio_path(info, 'T50kpcSUB'),
                    CHANDRA=info.get('chandra_path'))

    def record(self, info):
        cid = info['cluster_id']
        supplied = self.records.get(cid, info.get('physical_record'))
        if not isinstance(supplied, Mapping):
            raise ValueError(f'{cid}: provide matched_records[cluster_id] or info.physical_record')
        if supplied.get('cluster_id', cid) != cid:
            raise ValueError('Physical record belongs to another cluster')
        # Catalogue values only; labels and split memberships are not cache inputs.
        record = dict(cluster_id=cid)
        for name in ('ra_deg', 'dec_deg', 'redshift', 'r500_kpc', 'm500_msun'):
            value = supplied.get(name)
            if value is not None:
                value = float(value)
                if not np.isfinite(value):
                    value = None
            record[name] = value
        return record

    def load(self, info, size, *, enforce=True):
        pair, cache = self.store.get(self.sources(info), self.record(info), size,
                                      self.options.preprocessing_parameters())
        name = info.get('radio_source', 'RAW')
        selected = 'raw' if name == 'RAW' else 't50kcpsub'
        mask = (pair.common_radio_mask if self.options.radio_mask == 'intersection'
                else getattr(pair, selected+'_mask'))
        metadata = deepcopy(pair.metadata)
        reasons = [f'{n}={metadata["coverage"][n]:.6f} < {minimum:.6f}'
                   for n, minimum in self.options.coverage_thresholds.items()
                   if metadata['coverage'][n] < minimum]
        metadata.update(radio_source=name, radio_mask=self.options.radio_mask,
                        selected_radio_coverage=float(mask.mean()), cache=cache,
                        eligibility=dict(eligible=not reasons, reasons=reasons,
                            thresholds=dict(self.options.coverage_thresholds),
                            policy=self.options.coverage_policy))
        if enforce and reasons and self.options.coverage_policy == 'error':
            raise CoverageEligibilityError(metadata)
        # Always copy; consumers must never mutate a shared cached pair.
        return dict(raw=np.where(mask, getattr(pair, selected), np.nan).astype(np.float32),
                    raw_mask=mask.copy(), chandra=pair.chandra.copy(),
                    chandra_mask=pair.chandra_mask.copy(), wcs=pair.wcs.deepcopy(),
                    resolution_metadata=metadata)

    def item(self, dataset, index):
        """Historical raw/pair/multimodal schema plus radio mask and provenance."""
        source_index, variant = dataset._decode_index(index)
        key = dataset.keys[source_index]
        info = dataset.data_info[key]
        images = self.load(info, dataset.size)
        sample = dict(cluster_id=info['cluster_id'], key=key,
                      has_chandra=bool(info.get('chandra_path')),
                      resolution_metadata=sample_metadata(images['resolution_metadata']))
        for modality, name in [('radio', 'raw')] + ([] if dataset.mode == 'raw' else [('xray', 'chandra')]):
            image = images[name][None]
            mask = images[name+'_mask'][None]
            options = dataset.normalization.get(modality, {})
            bounds = dataset.normalization_bounds.get(modality)
            if options.get('scope', 'image') == 'train' and bounds is None:
                raise ValueError(f'Train normalization bounds missing for {modality}')
            tensor = torch.from_numpy(normalize_planes(image, mask, options, bounds))
            coverage = torch.from_numpy(mask.astype(np.float32))
            if dataset.augment:
                angle, flip, scale = decode_variant(variant, dataset.zoom)
                aligned = transform_aligned(torch.cat((tensor, tensor, coverage)),
                                            angle, flip, scale, dataset.zoom)
                tensor, coverage = aligned[1:2], aligned[2:3]
            sample[name], sample[name+'_mask'] = tensor, coverage
        if dataset.include_targets:
            sample['label'] = dataset._make_target(info)
            if dataset.task == 'rr_rh':
                sample['label_mask'] = torch.from_numpy(task_target(info, dataset.task, dataset.candidate_policy)[1])
            if dataset.task in SUBTYPE_TASKS:
                sample['annotations'] = '+'.join(sorted(labels_for(info)))
        return sample
