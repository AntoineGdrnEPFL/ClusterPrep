"""Inventaire et lecture des données FITS classifiées."""
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
import hashlib
import re

CLASS_NAMES = ("DE", "NDE")


class RadioInventory(dict):
    """Ordinary cluster mapping with a separate, non-sample discovery audit."""
    radio_audit = None


def get_data_info_dict(data_dir, class_names=None, *, legacy=None, return_audit=False):
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
    audit = discover_radio_products(data_info, data_dir)
    if legacy:
        data_info = discover_legacy(data_info, legacy.get("raw_root"),
                                    legacy.get("filename_pattern", "{cluster_id}_legacy.fits"))
    data_info = RadioInventory(data_info)
    data_info.radio_audit = audit
    return (data_info, audit) if return_audit else data_info

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


def validate_radio_source(source):
    """A product identifier, never a path (case sensitive)."""
    if not isinstance(source, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", source):
        raise ValueError("radio_source doit être un nom de produit sans chemin")
    return source


def radio_cluster_id(filename, product, canonical_ids):
    """Match exact RAW IDs or ID+product, optionally separated by _ or -.

    The concatenated convention is used by the PSZ2 survey and categoriser.
    Never strip arbitrary suffixes or silently choose between two valid IDs.
    """
    stem = Path(filename).stem
    candidates = {stem} & set(canonical_ids)
    for suffix in (product, "_" + product, "-" + product):
        if stem.endswith(suffix):
            candidate = stem[:-len(suffix)]
            if candidate in canonical_ids:
                candidates.add(candidate)
    if len(candidates) > 1:
        raise ValueError(f"Association radio ambiguë : {filename}: {sorted(candidates)}")
    return next(iter(candidates), None)


def discover_radio_products(data_info, data_dir):
    """Attach products without deriving any labels from their directories."""
    by_id = {info["cluster_id"]: info for info in data_info.values()}
    for info in data_info.values():
        info["radio_paths"] = {"RAW": info["raw_path"]}
    audit = {}
    # Other siblings (XMM, CHANDRA, optical, compact models) are not radio inputs.
    products = [p for p in sorted(Path(data_dir).iterdir()) if p.is_dir()
                and (p.name == "RAW" or re.fullmatch(r"(?:T|Blur)\d+[A-Za-z0-9]*|R-\d+(?:\.\d+)?", p.name))]
    for directory in products:
        product = directory.name
        report = {"present": [], "missing": [], "duplicates": [], "orphans": [], "naming_incompatibilities": []}
        seen = {}
        for path in sorted(directory.glob("*/*.fits")):
            cid = radio_cluster_id(path.name, product, by_id)
            if cid is None:
                report["orphans"].append(str(path))
                if not path.stem.endswith(product) and product != "RAW":
                    report["naming_incompatibilities"].append(str(path))
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if cid in seen:
                if seen[cid][0] != digest:
                    raise ValueError(f"Produit {product}, amas {cid}: contenus différents: {seen[cid][1]} et {path}")
                report["duplicates"].append(str(path))
            else:
                seen[cid] = (digest, str(path))
                by_id[cid]["radio_paths"][product] = str(path)
        report["present"] = sorted(seen)
        report["missing"] = sorted(set(by_id) - seen.keys())
        audit[product] = report
    for info in data_info.values():
        info["available_radio_sources"] = list(audit)
    return audit


def resolve_radio_path(info, radio_source=None):
    source = validate_radio_source(info.get("radio_source", "RAW") if radio_source is None else radio_source)
    path = info.get("raw_path") if source == "RAW" else info.get("radio_paths", {}).get(source)
    if not path:
        raise ValueError(f"Produit radio {source} manquant : {info.get('cluster_id', '?')}")
    return path


def select_radio_source(data_info, radio_source="RAW", missing_radio_policy="error"):
    """Experiment-local view; RAW identity and labels are never overwritten."""
    validate_radio_source(radio_source)
    if missing_radio_policy != "error":
        raise ValueError("missing_radio_policy: seule la politique stricte 'error' est supportée")
    available = {"RAW"}
    for info in data_info.values():
        available.update(info.get("available_radio_sources", []))
        available.update(info.get("radio_paths", {}))
    if radio_source not in available:
        raise ValueError(f"Produit radio inconnu {radio_source}; disponibles: {sorted(available)}")
    missing = [info['cluster_id'] for info in data_info.values()
               if not (info.get('raw_path') if radio_source == 'RAW' else info.get('radio_paths', {}).get(radio_source))]
    if missing:
        raise ValueError(f"Produit radio {radio_source} manquant pour {len(missing)} amas: {', '.join(sorted(missing))}")
    return {key: {**info, "radio_source": radio_source,
                  "radio_paths": {source: str(Path(path).resolve())
                                  for source, path in info.get("radio_paths", {"RAW": info["raw_path"]}).items()},
                  "active_radio_path": str(Path(resolve_radio_path(info, radio_source)).resolve())}
            for key, info in data_info.items()}


def radio_coverage(data_info):
    products = {"RAW"}
    for info in data_info.values():
        products.update(info.get("available_radio_sources", []))
        products.update(info.get("radio_paths", {}))
    return {source: {
        "present": sorted({i['cluster_id'] for i in data_info.values()
                           if (i.get('raw_path') if source == 'RAW' else i.get('radio_paths', {}).get(source))}),
        "missing": sorted({i['cluster_id'] for i in data_info.values()
                           if not (i.get('raw_path') if source == 'RAW' else i.get('radio_paths', {}).get(source))})}
        for source in sorted(products)}
