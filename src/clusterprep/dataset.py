"""Dataset PyTorch avec 24 variantes synchronisées par amas."""
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from .preprocessing import preprocess, normalize
from .processed import load_processed
from .scattering import load_scattering

class ClusterDataset(Dataset):
    ROTATIONS = tuple(range(0, 360, 30))
    FLIPS = (False, True)
    N_VARIANTS = len(ROTATIONS) * len(FLIPS)

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
        ):

        if mode not in {"raw", "pair", "multimodal"}:
            raise ValueError("mode doit être raw, pair ou multimodal")
        if not isinstance(size, int) or size < 1:
            raise ValueError("size doit être un entier positif")
        if len(set(class_names)) != len(class_names):
            raise ValueError("Classes dupliquées")
        self.data_info = data_info
        self.keys = list(keys)
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.mode = mode
        self.size = size
        self.augment = augment
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
        for key in self.keys:
            info = data_info[key]
            if info["class"] not in self.class_to_idx:
                raise ValueError(f"Classe inconnue : {info['class']}")
            if mode == "pair" and not info.get("chandra_path"):
                raise ValueError(f"Fichier CHANDRA manquant : {key}")

    def __len__(self):
        return len(self.keys) * (self.N_VARIANTS if self.augment else 1)

    def _decode_index(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return divmod(index, self.N_VARIANTS) if self.augment else (index, 0)

    def __getitem__(self, index):
        source_index, variant_index = self._decode_index(index)
        info = self.data_info[self.keys[source_index]]
        if source_index in self._cache:
            tensor = self._cache[source_index].clone()
        else:
            use_pair = self.mode != "raw" and bool(info.get("chandra_path"))
            if self.processed_dir is None:
                images = preprocess(info, mode="pair" if use_pair else "raw", size=self.size)
            else:
                images = load_processed(info, mode="pair" if use_pair else "raw",
                                        size=self.size, processed_dir=self.processed_dir)
            channels = [torch.from_numpy(normalize(images["raw"]))]
            if self.mode != "raw":
                mask = images.get("chandra_mask", np.zeros((self.size, self.size), dtype=bool))
                chandra = images.get("chandra", np.zeros((self.size, self.size), dtype=np.float32))
                channels.extend([torch.from_numpy(normalize(chandra, mask=mask)),
                                 torch.from_numpy(mask.astype(np.float32))])
            tensor = torch.stack(channels)
            if self.cache:
                self._cache[source_index] = tensor.clone()

        if self.augment:
            rotation_index, flip_index = divmod(variant_index, len(self.FLIPS))
            if self.FLIPS[flip_index]:
                tensor = TF.hflip(tensor)
            tensor = TF.rotate(
                tensor,
                angle=self.ROTATIONS[rotation_index],
                interpolation=InterpolationMode.NEAREST,
                expand=False,
                fill=0,
            )

        sample = {
            "raw": tensor[0:1],
            "label": torch.tensor(self.class_to_idx[info["class"]], dtype=torch.long),
            "cluster_id": info["cluster_id"],
            "key": self.keys[source_index],
            "has_chandra": bool(info.get("chandra_path")),
        }
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
        rotation_index, flip_index = divmod(variant_index, len(self.FLIPS))
        return {
            "cluster_id": info["cluster_id"],
            "class": info["class"],
            "raw_path": info["raw_path"],
            "chandra_path": info["chandra_path"],
            "rotation": self.ROTATIONS[rotation_index] if self.augment else 0,
            "flip_horizontal": self.FLIPS[flip_index] if self.augment else False,
        }
