"""Inventaire et lecture des données FITS classifiées."""
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

CLASS_NAMES = ("DE", "NDE")

def get_data_info_dict(data_dir: str | Path, class_names: Tuple[str, ...] = CLASS_NAMES) -> Dict[str, Dict[str, str]]:
    """
    Get a dictionary containing information about the data files in the specified directory.

    Args:
        data_dir (str | Path): The directory containing the data files.
        class_names (List[str]): A list of class names.
    Returns:
        Dict[str, Dict[str, str]]: A dictionary containing information about the data files.
    """
    data_dir = Path(data_dir)
    data_info = {}
    for class_name in class_names:
        class_dir_raw = data_dir / "RAW" / class_name
        class_dir_chandra = data_dir / "CHANDRA" / class_name
        if not class_dir_raw.exists():
            raise ValueError(f"Le répertoire {class_dir_raw} n'existe pas.")
        for file in sorted(class_dir_raw.glob("*.fits")):
            if file.name in data_info:
                raise ValueError(f"Nom FITS dupliqué entre classes : {file.name}")
            chandra_file = class_dir_chandra / str.replace(file.name, ".fits", "CHANDRA.fits")
            if not chandra_file.exists():
                chandra_file = None
            data_info[file.name] = {
                "cluster_id": file.stem,
                "raw_path": str(file),
                "chandra_path": str(chandra_file) if chandra_file else None,
                "name": file.name,
                "class": class_name,
                "size": str(file.stat().st_size)
            }
    return data_info

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
