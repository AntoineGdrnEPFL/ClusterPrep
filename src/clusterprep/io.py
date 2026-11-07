"""Inventaire et lecture des données FITS classifiées."""
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
import hashlib

CLASS_NAMES = ("DE", "NDE")

def get_data_info_dict(data_dir, class_names=None, *, legacy=None):
    """Une entrée par nom FITS, annotations fusionnées et sources vérifiées.

    Sans liste explicite, découvre les six catégories présentes.
    """
    from .labels import ANNOTATION_NAMES, labels_for
    data_dir = Path(data_dir)
    names = tuple(class_names) if class_names is not None else tuple(
        name for name in ANNOTATION_NAMES if (data_dir / "RAW" / name).is_dir())
    if not names:
        raise ValueError("Aucun répertoire RAW annoté trouvé")
    data_info = {}
    for class_name in names:
        directory = data_dir / "RAW" / class_name
        if not directory.is_dir():
            raise ValueError(f"Le répertoire {directory} n'existe pas.")
        for file in sorted(directory.glob("*.fits")):
            key = file.name  # interface historique ; cluster_id sans extension
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            info = data_info.setdefault(key, {
                "cluster_id": file.stem, "raw_path": str(file), "chandra_path": None,
                "name": file.name, "labels": set(), "size": str(file.stat().st_size), "hash": digest,
            })
            if info["hash"] != digest:
                raise ValueError(f"Fichier dupliqué avec contenu différent : {file}")
            info["labels"].add(class_name)
            labels_for(info)
    # CHANDRA peut se trouver dans une autre catégorie que le premier RAW rencontré.
    for directory in sorted((data_dir / "CHANDRA").glob("*")):
        for file in sorted(directory.glob("*CHANDRA.fits")):
            key = file.name.removesuffix("CHANDRA.fits") + ".fits"
            if key not in data_info:
                continue
            info = data_info[key]
            if info["chandra_path"] and Path(info["chandra_path"]).read_bytes() != file.read_bytes():
                raise ValueError(f"Sources CHANDRA contradictoires : {file}")
            info["chandra_path"] = info["chandra_path"] or str(file)
    return (discover_legacy(data_info, legacy.get("raw_root"),
                            legacy.get("filename_pattern", "{cluster_id}_legacy.fits"))
            if legacy else data_info)

def get_pairs(data_info: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Get a list of pairs of raw and Chandra data files based on their names.

    Args:
        data_info (Dict[str, Dict[str, str]]): A dictionary containing information about the data files.
    Returns:
        List[Dict[str, str]]: A list of pairs of raw and Chandra data files.
    """
    pairs = []
    for cluster_id, data in data_info.items():
        if data["chandra_path"] is not None:
            pairs.append(data)
    return pairs

def _find_celestial_image_hdu(hdul):
    """Trouve le premier HDU contenant une image et un WCS céleste 2D."""
    for hdu in hdul:
        if hdu.data is None or np.ndim(hdu.data) < 2:
            continue
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"'datfix' made the change.*",
                    category=FITSFixedWarning,
                )
                wcs = WCS(hdu.header).celestial
        except Exception:
            continue
        if wcs.has_celestial:
            return hdu, wcs
    raise ValueError("Aucun HDU image avec un WCS céleste valide n'a été trouvé.")

def image_to_2d(data):
    """Réduit les axes singleton d'un continuum radio et renvoie une image 2D."""
    image = np.squeeze(np.asarray(data))
    if image.ndim > 2:
        warnings.warn(
            f"Données de forme {image.shape} après squeeze : sélection du premier plan "
            "sur les axes non célestes. Vérifier ce choix pour un cube spectral."
        )
        while image.ndim > 2:
            image = image[0]
    if image.ndim != 2:
        raise ValueError(f"Impossible d'obtenir une image 2D (forme obtenue : {image.shape}).")
    return image.astype(float, copy=False)

def load_celestial_fits(path: str) -> Tuple[np.ndarray, fits.Header, WCS]:
    """Charge l'image 2D, le header de son HDU et son WCS céleste."""
    with fits.open(path, memmap=True) as hdul:
        hdu, wcs = _find_celestial_image_hdu(hdul)
        image = image_to_2d(hdu.data).copy()
        header = hdu.header.copy()
    return image, header, wcs

def read_fits_with_wcs(path):
    image, _, wcs = load_celestial_fits(path)
    return image.astype(np.float32), wcs


def discover_legacy(data_info, raw_root=None, filename_pattern="{cluster_id}_legacy.fits"):
    """Attach optional optical sources without changing IDs, labels or eligibility."""
    result = {key: dict(info) for key, info in data_info.items()}
    if raw_root is None:
        return result
    root = Path(raw_root)
    for info in result.values():
        cluster = info["cluster_id"]
        name = filename_pattern.format(cluster_id=cluster)
        path = root / cluster / name
        info["legacy_path"] = str(path) if path.is_file() else None
    return result


def load_optical_fits(path, bands=None, hdu=None):
    """Read selected planes, never silently discard a spectral/band axis."""
    with fits.open(path, memmap=False) as hdul:
        if hdu is None:
            selected, wcs = _find_celestial_image_hdu(hdul)
        else:
            selected = hdul[hdu]
            wcs = WCS(selected.header).celestial
        if not wcs.has_celestial:
            raise ValueError("Optical FITS requires celestial WCS")
        data = np.asarray(selected.data, dtype=np.float32)
        if data.ndim == 2:
            data = data[None]
        if data.ndim != 3:
            raise ValueError(f"Expected [bands, y, x] optical image, got {data.shape}")
        header = selected.header.copy()
        names = [str(header.get(f"BAND{i}", i)).strip() for i in range(len(data))]
        indices = list(range(len(data))) if bands is None else [
            names.index(b) if isinstance(b, str) else b for b in bands]
        if any(i < 0 or i >= len(data) for i in indices):
            raise ValueError("Optical band index out of range")
        return data[indices].copy(), header, wcs
