"""Préparation alignée sur la grille radio, sans dépendance à dcreclass."""
import numpy as np
from reproject import reproject_interp
from .io import read_fits_with_wcs

def block_average(image, factor):
    """Moyenne des valeurs finies dans des blocs factor × factor.

    Un bloc entièrement sans données reste NaN.
    """
    image = np.asarray(image)
    if image.ndim != 2 or not isinstance(factor, int) or factor < 1:
        raise ValueError("Image 2D et facteur entier positif attendus")
    height, width = image.shape
    if height % factor or width % factor:
        raise ValueError("Dimensions non divisibles par factor")
    blocks = image.reshape(height // factor, factor, width // factor, factor)
    valid = np.isfinite(blocks)
    count = valid.sum(axis=(1, 3))
    total = np.where(valid, blocks, 0).sum(axis=(1, 3), dtype=np.float64)
    result = np.full(count.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=result, where=count > 0)
    return result


def centered_raw_window(shape, size, factor):
    """Crop centré, aligné sur les blocs d'une réduction de RAW entier."""
    height, width = shape
    blocks_y, blocks_x = height // factor, width // factor
    if blocks_y < size or blocks_x < size:
        raise ValueError("RAW trop petit pour ce facteur")
    y0 = ((blocks_y - size) // 2) * factor
    x0 = ((blocks_x - size) // 2) * factor
    side = size * factor
    return slice(y0, y0 + side), slice(x0, x0 + side)


def preprocess(info, mode, size=128):
    """Prépare des images alignées de taille (size, size).

    Toute paire disposant d'un fichier CHANDRA est conservée, même en
    l'absence totale de recouvrement. Le masque signale les pixels CHANDRA
    finis après reprojection et moyenne par blocs.
    """
    if mode not in {"raw", "pair"}:
        raise ValueError("mode doit être 'raw' ou 'pair'")
    if not isinstance(size, int) or size < 1:
        raise ValueError("size doit être positif")

    raw, raw_wcs = read_fits_with_wcs(info["raw_path"])
    ratio = min(raw.shape) // size
    if ratio < 1:
        raise ValueError(f"RAW trop petit : {info['cluster_id']}")
    factor = 1 << (ratio.bit_length() - 1)
    window = centered_raw_window(raw.shape, size, factor)
    raw_window = raw[window]
    raw_small = block_average(raw_window, factor)
    reduced_window = tuple(slice(s.start, s.stop, factor) for s in window)
    result = {"raw": raw_small, "factor": factor, "wcs": raw_wcs.slice(reduced_window)}

    if mode == "pair":
        if not info.get("chandra_path"):
            raise ValueError(f"Fichier CHANDRA manquant : {info['cluster_id']}")
        chandra, chandra_wcs = read_fits_with_wcs(info["chandra_path"])
        window_wcs = raw_wcs.slice(window)
        chandra_on_raw, footprint = reproject_interp(
            (chandra, chandra_wcs),
            window_wcs,
            shape_out=raw_window.shape,
            order="bilinear",
        )
        chandra_on_raw[footprint <= 0] = np.nan
        chandra_small = block_average(chandra_on_raw, factor)
        chandra_mask = np.isfinite(chandra_small)  # (size, size), pas taille avant réduction
        result.update(chandra=chandra_small, chandra_mask=chandra_mask)

    return result


def normalize(image, mask=None, percentile_lo=30, percentile_hi=99, alpha=10.0):
    """Percentiles par image puis étirement asinh, comme dans dcreclass."""
    if not 0 <= percentile_lo < percentile_hi <= 100 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Percentiles ou alpha invalides")
    image = np.asarray(image)
    if mask is not None and np.shape(mask) != image.shape:
        raise ValueError("Le masque doit avoir la même forme que l’image")
    valid = np.isfinite(image)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    output = np.zeros(image.shape, dtype=np.float32)
    if not valid.any():
        return output
    values = image[valid].astype(np.float64)
    p_low = np.quantile(values, percentile_lo / 100)
    p_high = np.quantile(values, percentile_hi / 100)
    scaled = np.clip((values - p_low) / (p_high - p_low + 1e-6), 0, 1)
    output[valid] = (np.arcsinh(alpha * scaled) / np.arcsinh(alpha)).astype(np.float32)
    return output

