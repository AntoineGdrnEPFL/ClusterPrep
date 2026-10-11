"""Précalcul et cache des coefficients de scattering."""

import json
import hashlib
from .processed import CACHE_VERSION
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch

from .progress import progress, message

SCATTERING_CACHE_VERSION = 2


def _source_signature(info, mode):
    """Identité des fichiers sources pour invalider le cache si nécessaire."""
    result = {}

    for name in (("raw_path",) if mode == "raw" else ("raw_path", "chandra_path")):
        path = info.get(name)

        if path is None:
            result[name] = None
            continue

        path = Path(path).resolve()
        stat = path.stat()

        result[name] = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }

    return result


def scattering_identity(
    info,
    mode,
    size,
    J,
    L,
    max_order,
):
    return {
        "version": SCATTERING_CACHE_VERSION,
        "preprocessing_version": CACHE_VERSION,
        "cluster_id": info["cluster_id"],
        "mode": mode,
        "size": size,
        "J": J,
        "L": L,
        "max_order": max_order,
        # Important : le scattering est calculé APRES cette normalisation.
        "normalization": {
            "percentile_lo": 30,
            "percentile_hi": 99,
            "alpha": 10.0,
        },
        # Important également : les 24 variantes sont précalculées.
        "augmentation": {
            "rotations": list(range(0, 360, 30)),
            "horizontal_flip": [False, True],
        },
        "sources": _source_signature(info, mode),
    }


def scattering_paths(
    info,
    mode,
    size,
    J,
    L,
    max_order,
    scattering_dir,
):
    cluster = quote(str(info["cluster_id"]), safe="+")
    if cluster in {"", ".", ".."}:
        raise ValueError("cluster_id invalide")
    root = Path(scattering_dir) / cluster

    stem = (
        f"{cluster}_scattering_"
        f"{mode}_size{size}_"
        f"J{J}_L{L}_O{max_order}_"
        f"v{SCATTERING_CACHE_VERSION}"
    )

    return {
        "data": root / f"{stem}.npy",
        "metadata": root / f"{stem}.json",
    }


def load_scattering(
    info,
    mode,
    size,
    scattering_dir,
    J=2,
    L=8,
    max_order=2,
):
    """
    Charge le cache sous forme de memmap.

    Shape :
        [N_variants, C, K, Hs, Ws]
    """

    paths = scattering_paths(
        info,
        mode,
        size,
        J,
        L,
        max_order,
        scattering_dir,
    )

    if not paths["data"].exists() or not paths["metadata"].exists():
        raise FileNotFoundError(f"Cache scattering absent pour {info['cluster_id']}")

    metadata = json.loads(paths["metadata"].read_text())

    expected = scattering_identity(
        info,
        mode,
        size,
        J,
        L,
        max_order,
    )

    if metadata["identity"] != expected:
        raise ValueError(f"Cache scattering invalide pour {info['cluster_id']}")

    coefficients = np.load(
        paths["data"],
        mmap_mode="r",
        allow_pickle=False,
    )

    K = 1 + J * L + (L * L * J * (J - 1) // 2 if max_order == 2 else 0)
    expected_shape = (24, 1 if mode == "raw" else 3, K, size // 2**J, size // 2**J)
    if tuple(metadata["shape"]) != expected_shape or coefficients.dtype != np.float32:
        raise ValueError("Métadonnées scattering invalides")
    if hashlib.sha256(paths["data"].read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Checksum scattering invalide")
    if not np.isfinite(coefficients).all():
        raise ValueError("Coefficients non finis")

    if coefficients.shape != expected_shape:
        raise ValueError(
            f"Shape scattering invalide : " f"{coefficients.shape} != {expected_shape}"
        )

    return coefficients


def _write_cache(paths, coefficients, identity):
    paths["data"].parent.mkdir(parents=True, exist_ok=True)

    # ---------- coefficients ----------
    fd, temporary = tempfile.mkstemp(
        prefix=".scattering-",
        suffix=".npy",
        dir=paths["data"].parent,
    )
    os.close(fd)

    try:
        np.save(
            temporary,
            coefficients.astype(np.float32, copy=False),
            allow_pickle=False,
        )

        os.replace(temporary, paths["data"])

    finally:
        Path(temporary).unlink(missing_ok=True)

    # ---------- metadata ----------
    metadata = {
        "identity": identity,
        "shape": list(coefficients.shape),
        "dtype": "float32",
        "sha256": hashlib.sha256(paths["data"].read_bytes()).hexdigest(),
    }

    fd, name = tempfile.mkstemp(prefix=".scattering-", suffix=".json", dir=paths["metadata"].parent)
    temporary_metadata = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
        temporary_metadata.replace(paths["metadata"])
    finally:
        temporary_metadata.unlink(missing_ok=True)



def prepare_scattering(
    data_info,
    keys,
    class_names,
    mode="raw",
    size=128,
    processed_dir="processed",
    scattering_dir="processed/scattering",
    J=2,
    L=8,
    max_order=2,
    device="cpu",
    verbose=True,
    force=False,
    task="de_nde",
    candidate_policy="positive",
):
    """
    Précalcule les 24 scattering de chaque amas.

    Les images passent exactement par ClusterDataset :
        preprocessing
        -> normalisation
        -> rotation/flip
        -> scattering
    """

    from kymatio.torch import Scattering2D

    # import local pour éviter une dépendance circulaire
    from .dataset import ClusterDataset

    unique = {}
    for key in keys:
        cid = data_info[key]["cluster_id"]
        if cid in unique and scattering_identity(data_info[unique[cid]], mode, size, J, L, max_order) != scattering_identity(data_info[key], mode, size, J, L, max_order):
            raise ValueError(f"Sources contradictoires pour {cid}")
        unique.setdefault(cid, key)
    keys = list(unique.values())
    device = torch.device(device)

    scattering = Scattering2D(
        J=J,
        shape=(size, size),
        L=L,
        max_order=max_order,
    ).to(device)

    counts = {
        "created": 0,
        "reused": 0,
    }

    message(
        f"Cache scattering : {len(keys)} amas, "
        f"J={J}, L={L}, order={max_order}, device={device}",
        verbose,
    )

    with progress(keys, "Scattering", verbose) as bar:

        for key in bar:
            info = data_info[key]

            paths = scattering_paths(
                info,
                mode,
                size,
                J,
                L,
                max_order,
                scattering_dir,
            )

            if not force:
                try:
                    load_scattering(
                        info,
                        mode,
                        size,
                        scattering_dir,
                        J=J,
                        L=L,
                        max_order=max_order,
                    )

                    counts["reused"] += 1
                    continue

                except (FileNotFoundError, ValueError, OSError, KeyError):
                    pass

            # On utilise volontairement le Dataset existant :
            # c'est lui qui définit normalisation + augmentation.
            dataset = ClusterDataset(
                data_info,
                [key],
                class_names,
                mode=mode,
                size=size,
                augment=True,
                cache=True,
                processed_dir=processed_dir,
                scattering_dir=None,
                include_targets=False,
            )

            initial_identity = scattering_identity(info, mode, size, J, L, max_order)
            variants = []

            with torch.inference_mode():

                for index in range(dataset.N_VARIANTS):

                    sample = dataset[index]

                    names = (
                        ("raw",)
                        if mode == "raw"
                        else ("raw", "chandra", "chandra_mask")
                    )

                    x = torch.cat(
                        [sample[name] for name in names],
                        dim=0,
                    )

                    # [C,H,W] -> [1,C,H,W]
                    x = x.unsqueeze(0).to(device)

                    # [1,C,K,Hs,Ws]
                    coeff = scattering(x)

                    # -> [C,K,Hs,Ws]
                    variants.append(coeff.squeeze(0).cpu().numpy())

            coefficients = np.stack(
                variants,
                axis=0,
            ).astype(np.float32)

            identity = scattering_identity(
                info,
                mode,
                size,
                J,
                L,
                max_order,
            )

            if identity != initial_identity:
                raise RuntimeError("Les sources ont changé pendant le scattering")
            _write_cache(
                paths,
                coefficients,
                identity,
            )

            counts["created"] += 1

            bar.set_postfix(
                créés=counts["created"],
                réutilisés=counts["reused"],
                refresh=False,
            )

    return counts
