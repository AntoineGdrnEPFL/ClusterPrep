"""Dataset PyTorch avec variantes géométriques synchronisées par amas."""

import numpy as np
import torch
from torch.utils.data import Dataset
from .io import resolve_radio_path
from .preprocessing import preprocess, normalize
from .processed import load_processed
from .scattering import load_scattering
from .labels import labels_for, task_target, merge_cluster_labels, BINARY_TASKS, SUBTYPE_TASKS, TASK_NAMES
from .processed import _identity
from .augmentation import (ROTATIONS, FLIPS, GEOMETRIC_VARIANTS, validate_zoom,
                           variant_count, decode_variant, transform_aligned, zoom_identity)
from .radio_noise import validate_radio_noise, prepare_noise_context, add_radio_noise


class ClusterDataset(Dataset):
    ROTATIONS = ROTATIONS
    FLIPS = FLIPS
    N_VARIANTS = GEOMETRIC_VARIANTS

    def __init__(
        self,
        data_info,
        keys,
        class_names,
        mode="raw",
        size=128,
        augment=False,
        cache=False,
        processed_dir="processed",
        scattering_dir=None,
        scattering_params=None,
        task="de_nde",
        candidate_policy="positive",
        include_targets=True,
        radio_noise=None,
        training=False,
        zoom=None,
        modalities=None,
        missing_modality_policy="optional",
        legacy=None,
        include_masks=True,
        normalization=None,
        normalization_bounds=None,
        radio_source=None,
        missing_radio_policy="error",
    ):

        from .modalities import validate_modalities, PATHS
        from .io import discover_legacy, select_radio_source
        if radio_source is not None:
            data_info = select_radio_source(data_info, radio_source, missing_radio_policy)
        validate_modalities(modalities, missing_modality_policy, legacy, include_masks)
        self.modalities = modalities
        self.legacy = legacy or {}
        self.include_masks = include_masks
        self.normalization = normalization or {}
        self.normalization_bounds = normalization_bounds or {}
        if modalities is not None:
            if scattering_dir is not None:
                raise ValueError("Explicit modalities do not support precomputed scattering")
            data_info = discover_legacy(data_info, self.legacy.get("raw_root"),
                                       self.legacy.get("filename_pattern", "{cluster_id}_legacy.fits"))
            if missing_modality_policy == "required":
                keys = [k for k in keys if all(data_info[k].get(PATHS[m]) for m in modalities)]
            if augment and not training:
                raise ValueError("Explicit modality augmentation is restricted to training")
        if mode not in {"raw", "pair", "multimodal"}:
            raise ValueError("mode doit être raw, pair ou multimodal")
        if not isinstance(size, int) or size < 1:
            raise ValueError("size doit être un entier positif")
        if len(set(class_names)) != len(class_names):
            raise ValueError("Classes dupliquées")
        if task not in TASK_NAMES or candidate_policy not in {
            "positive",
            "ignore",
        }:
            raise ValueError("task ou candidate_policy invalide")
        self.radio_noise = validate_radio_noise(radio_noise)
        self.noise_enabled = training and self.radio_noise["enabled"]
        if self.noise_enabled and scattering_dir is not None:
            raise ValueError("radio_noise incompatible avec le scattering précalculé")
        self._noise_cache = {}
        self.include_targets = include_targets
        data_info = merge_cluster_labels(data_info) if include_targets else data_info
        self.data_info = data_info
        self.keys = list(keys)
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.mode = mode
        self.size = size
        self.augment = augment
        self.zoom = validate_zoom(zoom)
        self.N_VARIANTS = variant_count(self.zoom)
        self.processed_dir = processed_dir
        self.cache = cache
        self._cache = {}
        self.scattering_dir = scattering_dir
        self.scattering_params = scattering_params or {
            "J": 2,
            "L": 8,
            "max_order": 2,
        }
        self._scattering_cache = {}
        self.task = task
        self.candidate_policy = candidate_policy
        for key in self.keys:
            info = data_info[key]
            if not include_targets:
                if modalities is None and mode == "pair" and not info.get("chandra_path"):
                    raise ValueError(f"Fichier CHANDRA manquant : {key}")
                continue
            labels = labels_for(info)
            if self.task == "de_nde":
                if not {"DE", "NDE"}.issubset(self.class_to_idx):
                    raise ValueError("La tâche de_nde nécessite les classes DE et NDE")
                if not labels.intersection({"DE", "NDE"}):
                    raise ValueError(f"Cluster non éligible DE/NDE : {key}")
                if {"DE", "NDE"}.issubset(labels):
                    raise ValueError(f"Cluster ambigu DE/NDE : {key}")
            elif self.task in SUBTYPE_TASKS:
                task_target(info, self.task, candidate_policy)
            elif self.task == "rr_rh":
                if not ("NDE" in labels or labels & {"RR", "RH", "cRR", "cRH"}):
                    raise ValueError(f"Cluster non éligible RR/RH : {key}")
            else:
                raise ValueError(f"Tâche inconnue : {self.task}")
            if modalities is None and mode == "pair" and not info.get("chandra_path"):
                raise ValueError(f"Fichier CHANDRA manquant : {key}")

    def __len__(self):
        return len(self.keys) * (self.N_VARIANTS if self.augment else 1)

    def _decode_index(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return divmod(index, self.N_VARIANTS) if self.augment else (index, 0)

    def _make_target(self, info):
        target, _ = task_target(info, self.task, self.candidate_policy, list(self.class_to_idx))
        return torch.tensor(target, dtype=torch.long if self.task in BINARY_TASKS else torch.float32)

    def __getitem__(self, index):
        if self.modalities is not None:
            return self._modality_item(index)
        source_index, variant_index = self._decode_index(index)
        info = self.data_info[self.keys[source_index]]
        use_pair = self.mode != "raw" and bool(info.get("chandra_path"))
        identity = _identity(info, "pair" if use_pair else "raw", self.size)
        cache_key = repr((identity, zoom_identity(self.zoom) if self.augment else None))
        if getattr(self, "_identities", {}).get(source_index) != cache_key:
            self._cache.pop(source_index, None)
            self._noise_cache.pop(source_index, None)
            self._scattering_cache.pop(source_index, None)
        if not hasattr(self, "_identities"):
            self._identities = {}
        self._identities[source_index] = cache_key
        if self.noise_enabled:
            # Cache deterministic linear data; draw fresh noise before geometry.
            if source_index not in self._noise_cache:
                images = (preprocess(info, "pair" if use_pair else "raw", self.size)
                          if self.processed_dir is None else
                          load_processed(info, "pair" if use_pair else "raw", self.size, self.processed_dir))
                context = prepare_noise_context(images, info, self.processed_dir)
                clean = images["raw"]
                bounds = np.quantile(clean[np.isfinite(clean)].astype(float), [.30, .99])
                self._noise_cache[source_index] = images, context, bounds
            images, context, bounds = self._noise_cache[source_index]
            noisy = add_radio_noise(images["raw"], context, self.radio_noise)
            channels = [torch.from_numpy(normalize(noisy, bounds=bounds))]
            if self.mode != "raw":
                mask = images.get("chandra_mask", np.zeros((self.size, self.size), dtype=bool))
                chandra = images.get("chandra", np.zeros((self.size, self.size), dtype=np.float32))
                channels.extend([torch.from_numpy(normalize(chandra, mask=mask)),
                                 torch.from_numpy(mask.astype(np.float32))])
            tensor = torch.stack(channels)
        elif source_index in self._cache:
            tensor = self._cache[source_index].clone()
        else:
            use_pair = self.mode != "raw" and bool(info.get("chandra_path"))
            if self.processed_dir is None:
                images = preprocess(
                    info, mode="pair" if use_pair else "raw", size=self.size
                )
            else:
                images = load_processed(
                    info,
                    mode="pair" if use_pair else "raw",
                    size=self.size,
                    processed_dir=self.processed_dir,
                )
            channels = [torch.from_numpy(normalize(images["raw"]))]
            if self.mode != "raw":
                mask = images.get(
                    "chandra_mask", np.zeros((self.size, self.size), dtype=bool)
                )
                chandra = images.get(
                    "chandra", np.zeros((self.size, self.size), dtype=np.float32)
                )
                channels.extend(
                    [
                        torch.from_numpy(normalize(chandra, mask=mask)),
                        torch.from_numpy(mask.astype(np.float32)),
                    ]
                )
            tensor = torch.stack(channels)
            if self.cache:
                self._cache[source_index] = tensor.clone()

        if self.augment:
            angle, flip, scale = decode_variant(variant_index, self.zoom)
            tensor = transform_aligned(tensor, angle, flip, scale, self.zoom)

        sample = {
            "raw": tensor[0:1],
            "cluster_id": info["cluster_id"],
            "key": self.keys[source_index],
            "has_chandra": bool(info.get("chandra_path")),
        }
        if self.include_targets and self.task in SUBTYPE_TASKS:
            sample["annotations"] = "+".join(sorted(labels_for(info)))
        if self.include_targets:
            sample["label"] = self._make_target(info)
        if self.include_targets and self.task == "rr_rh":
            sample["label_mask"] = torch.from_numpy(
                task_target(info, self.task, self.candidate_policy)[1]
            )
        if self.mode != "raw":
            sample["chandra"] = torch.nan_to_num(tensor[1:2], nan=0.0)
            sample["chandra_mask"] = tensor[2:3]

        if self.scattering_dir is not None:

            if source_index not in self._scattering_cache:

                coefficients = load_scattering(
                    info,
                    mode=self.mode,
                    size=self.size,
                    scattering_dir=self.scattering_dir,
                    zoom=self.zoom if self.augment else None,
                    **self.scattering_params,
                )

                self._scattering_cache[source_index] = coefficients

            coefficients = self._scattering_cache[source_index]

            # validation/test -> variante 0
            # train augmenté   -> variante correspondante
            scatter_index = variant_index if self.augment else 0

            # copy nécessaire car le np.memmap est read-only
            sample["scattering"] = torch.from_numpy(
                np.asarray(coefficients[scatter_index]).copy()
            )
        return sample

    def get_item_info(self, index):
        source_index, variant_index = self._decode_index(index)
        info = self.data_info[self.keys[source_index]]
        angle, flip, scale = decode_variant(variant_index, self.zoom) if self.augment else (0, False, 1.0)
        return {
            "cluster_id": info["cluster_id"],
            "labels": sorted(labels_for(info)),
            "task": self.task,
            "target": self._make_target(info).tolist(),
            "raw_path": info["raw_path"],
            "radio_source": info.get("radio_source", "RAW"),
            "active_radio_path": resolve_radio_path(info),
            "chandra_path": info.get("chandra_path"),
            "scattering_path": info.get("scattering_path"),
            "legacy_path": info.get("legacy_path"),
            "candidate_policy": self.candidate_policy,
            "rotation": angle,
            "flip_horizontal": flip,
            "zoom_factor": scale,
            "source_index": source_index,
            "variant_index": variant_index,
        }


    def _modality_images(self, info):
        from .processed import load_processed_optical
        from .preprocessing import preprocess_optical
        paired = "xray" in self.modalities and bool(info.get("chandra_path"))
        mode = "pair" if paired else "raw"
        images = (preprocess(info, mode, self.size) if self.processed_dir is None else
                  load_processed(info, mode, self.size, self.processed_dir))
        if "legacy" in self.modalities and info.get("legacy_path"):
            options = dict(bands=self.legacy.get("bands", ["g", "r", "z"]), hdu=self.legacy.get("hdu"))
            optical = (preprocess_optical(info, self.size, **options) if self.processed_dir is None else
                       load_processed_optical(info, self.size, self.processed_dir, **options))
            images.update(optical)
        return images

    def _modality_item(self, index):
        from .modalities import layout, NAMES, PATHS, normalize_planes
        source_index, variant = self._decode_index(index)
        key = self.keys[source_index]
        info = self.data_info[key]
        # Disk cache validates source provenance on every read.
        images = self._modality_images(info)
        sample = {"cluster_id": info["cluster_id"], "key": key,
                  "has_chandra": bool(info.get("chandra_path")),
                  "has_legacy": bool(info.get("legacy_path"))}
        stacked = []
        availability = []
        for modality, count, with_mask in layout(self.modalities, self.legacy, self.include_masks):
            name = NAMES[modality]
            image = np.asarray(images.get(name, np.zeros((count, self.size, self.size))), dtype=np.float32)
            if image.ndim == 2:
                image = image[None]
            mask = np.asarray(images.get(name + "_mask", np.isfinite(image)), dtype=bool).reshape(image.shape)
            if not info.get(PATHS[modality]):
                mask[:] = False
            if image.shape != (count, self.size, self.size):
                raise ValueError(f"Unexpected {modality} shape: {image.shape}")
            options = (self.legacy.get("normalization", {}) if modality == "legacy" else
                       self.normalization.get(modality, {}))
            bounds = self.normalization_bounds.get(modality)
            if options.get("scope", "image") == "train" and bounds is None:
                raise ValueError(f"Train normalization bounds missing for {modality}")
            if modality == "radio" and self.noise_enabled:
                context = prepare_noise_context(images, info, self.processed_dir)
                if bounds is None:
                    bounds = [np.quantile(image[np.isfinite(image)], [.30, .99])]
                image = add_radio_noise(image[0], context, self.radio_noise)[None]
            image = normalize_planes(image, mask, options, bounds)
            tensor, coverage = torch.from_numpy(image), torch.from_numpy(mask.astype(np.float32))
            if self.augment:
                angle, flip, scale = decode_variant(variant, self.zoom)
                # Existing validity-weighted geometry, same grid for every band and mask.
                transformed = [transform_aligned(torch.stack((tensor[i], tensor[i], coverage[i])),
                                                 angle, flip, scale, self.zoom) for i in range(count)]
                tensor = torch.stack([t[1] for t in transformed])
                coverage = torch.stack([t[2] for t in transformed])
            sample[name] = tensor
            sample[name + "_mask"] = coverage
            stacked.append(tensor)
            if with_mask:
                stacked.append(coverage)
            availability.append(bool(info.get(PATHS[modality])))
        sample["image"] = torch.cat(stacked)
        sample["availability"] = torch.tensor(availability, dtype=torch.bool)
        if self.include_targets:
            sample["label"] = self._make_target(info)
            if self.task == "rr_rh":
                sample["label_mask"] = torch.from_numpy(task_target(info, self.task, self.candidate_policy)[1])
            if self.task in SUBTYPE_TASKS:
                sample["annotations"] = "+".join(sorted(labels_for(info)))
        return sample
