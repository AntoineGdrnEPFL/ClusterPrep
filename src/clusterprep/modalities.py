"""Opt-in modality layout and normalization; historical layouts remain unchanged."""
import numpy as np

PATHS = {"radio": "raw_path", "xray": "chandra_path", "legacy": "legacy_path"}
NAMES = {"radio": "raw", "xray": "chandra", "legacy": "legacy"}


def validate_modalities(modalities, policy="optional", legacy=None, masks=True):
    if modalities is None:
        return
    if (not isinstance(modalities, (list, tuple)) or not modalities
            or len(set(modalities)) != len(modalities) or set(modalities) - PATHS.keys()):
        raise ValueError("modalities: unique nonempty selection of radio, xray, legacy")
    if policy not in {"required", "optional"}:
        raise ValueError("missing_modality_policy: required or optional")
    if policy == "optional" and not masks:
        raise ValueError("Optional modalities require coverage mask channels")
    options = legacy or {}
    if set(options) - {"raw_root", "filename_pattern", "bands", "hdu", "normalization"}:
        raise ValueError("Unknown legacy option")
    bands = options.get("bands", ["g", "r", "z"])
    if not bands or len(set(bands)) != len(bands) or not all(isinstance(b, (str, int)) for b in bands):
        raise ValueError("legacy.bands must select unique band names or indices")


def layout(modalities, legacy=None, masks=True):
    result = []
    for modality in modalities:
        channels = len((legacy or {}).get("bands", ["g", "r", "z"])) if modality == "legacy" else 1
        result.append((modality, channels, bool(masks and modality != "radio")))
    return result


def channel_count(modalities, legacy=None, masks=True):
    return sum(c * (2 if m else 1) for _, c, m in layout(modalities, legacy, masks))


def normalize_planes(image, mask, options=None, bounds=None):
    from .preprocessing import normalize
    options = options or {}
    if set(options) - {"scope", "method", "lower_percentile", "upper_percentile", "alpha"}:
        raise ValueError("Unknown normalization option")
    method = options.get("method", "asinh_percentile")
    if method not in {"percentile", "asinh_percentile"}:
        raise ValueError("normalization.method: percentile or asinh_percentile")
    if options.get("scope", "image") not in {"image", "train"}:
        raise ValueError("normalization.scope: image or train")
    lo, hi = options.get("lower_percentile", 30), options.get("upper_percentile", 99)
    alpha = options.get("alpha", 10)
    result = []
    for index, plane in enumerate(image):
        value = normalize(plane, mask[index], lo, hi, alpha,
                          None if bounds is None else bounds[index])
        if method == "percentile":
            value = (np.sinh(value * np.arcsinh(alpha)) / alpha).astype(np.float32)
        result.append(value)
    return np.stack(result)


def fit_normalization(dataset):
    """Fit only on caller-supplied training keys, before augmentation/noise."""
    pending = {}
    for modality in dataset.modalities:
        options = (dataset.legacy.get("normalization", {}) if modality == "legacy" else
                   dataset.normalization.get(modality, {}))
        if options.get("scope", "image") == "train":
            pending[modality] = options
    if not pending:
        return {}
    values = {m: [[] for _ in range(next(c for n, c, _ in layout(dataset.modalities, dataset.legacy) if n == m))]
              for m in pending}
    for key in dataset.keys:
        images = dataset._modality_images(dataset.data_info[key])
        for modality in pending:
            name = NAMES[modality]
            if name not in images:
                continue
            planes = np.atleast_3d(images[name]) if images[name].ndim == 3 else images[name][None]
            mask = np.asarray(images.get(name + "_mask", np.isfinite(planes))).reshape(planes.shape)
            for index, plane in enumerate(planes):
                values[modality][index].append(plane[np.isfinite(plane) & mask[index]])
    bounds = {}
    for modality, options in pending.items():
        bounds[modality] = []
        for parts in values[modality]:
            pixels = np.concatenate(parts) if parts else np.array([])
            if not pixels.size:
                raise ValueError(f"No covered training pixels for {modality} normalization")
            bounds[modality].append(np.percentile(pixels, [options.get("lower_percentile", 30),
                                                          options.get("upper_percentile", 99)]).tolist())
    return bounds
