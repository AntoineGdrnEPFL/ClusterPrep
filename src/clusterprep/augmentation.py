"""Deterministic, centred geometry on already aligned, normalized images."""
import math

import torch
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode, functional as TF


ROTATIONS = tuple(range(0, 360, 30))
FLIPS = (False, True)
GEOMETRIC_VARIANTS = len(ROTATIONS) * len(FLIPS)
ZOOM_VERSION = 1


def validate_zoom(config=None):
    defaults = dict(enabled=False, factors=[1.0], interpolation="bilinear",
                    padding_mode="constant", fill_value=0.0)
    if config is None:
        return defaults
    if not isinstance(config, dict) or set(config) - set(defaults):
        raise ValueError("augmentation.zoom: configuration ou clés invalides")
    result = {**defaults, **config}
    if not isinstance(result["enabled"], bool):
        raise ValueError("zoom.enabled doit être booléen")
    if result["enabled"] and "factors" not in config:
        raise ValueError("zoom.factors doit être défini explicitement")
    factors = result["factors"]
    if (not isinstance(factors, (list, tuple)) or not factors
            or any(isinstance(s, bool) or not isinstance(s, (int, float))
                   or not math.isfinite(s) or s <= 0 for s in factors)):
        raise ValueError("zoom.factors: liste non vide de nombres finis strictement positifs")
    if result["interpolation"] not in ("bilinear", "nearest"):
        raise ValueError("zoom.interpolation: bilinear ou nearest")
    if result["padding_mode"] not in ("constant", "reflect", "replicate"):
        raise ValueError("zoom.padding_mode: constant, reflect ou replicate")
    fill = result["fill_value"]
    if isinstance(fill, bool) or not isinstance(fill, (int, float)) or not math.isfinite(fill):
        raise ValueError("zoom.fill_value doit être un nombre fini")
    return {**result, "factors": [float(s) for s in factors], "fill_value": float(fill)}


def zoom_identity(config=None):
    """Disabled zoom keeps the historical cache identity and file names."""
    zoom = validate_zoom(config)
    return {"version": ZOOM_VERSION, **zoom} if zoom["enabled"] else None


def variant_count(config=None):
    zoom = validate_zoom(config)
    return GEOMETRIC_VARIANTS * (len(zoom["factors"]) if zoom["enabled"] else 1)


def decode_variant(index, config=None):
    zoom = validate_zoom(config)
    zoom_index, geometry_index = divmod(index, GEOMETRIC_VARIANTS)
    rotation_index, flip_index = divmod(geometry_index, len(FLIPS))
    scale = zoom["factors"][zoom_index] if zoom["enabled"] else 1.0
    return ROTATIONS[rotation_index], FLIPS[flip_index], scale


def transform_aligned(tensor, angle=0, flip=False, scale=1.0, zoom=None):
    """Transform [radio, optional Chandra, mask] with one common inverse grid.

    Scale one uses the exact historical rotation/flip path. Otherwise Chandra
    is interpolated with validity weights; its coverage never uses padding.
    """
    zoom = validate_zoom(zoom)
    if not zoom["enabled"] or scale == 1.0:
        if flip:
            tensor = TF.hflip(tensor)
        return TF.rotate(tensor, angle=angle, interpolation=InterpolationMode.NEAREST,
                         expand=False, fill=0)

    height, width = tensor.shape[-2:]
    radians = math.radians(angle)
    c, s = math.cos(radians) / scale, math.sin(radians) / scale
    sign = -1 if flip else 1
    # Inverse of flip followed by counterclockwise rotation and scaling.
    theta = tensor.new_tensor([[sign*c, -sign*s*height/width, 0],
                               [s*width/height, c, 0]])[None]
    grid = F.affine_grid(theta, (1, 1, height, width), align_corners=False)

    def sample(values, mode, padding="zeros"):
        return F.grid_sample(values[None], grid, mode=mode,
                             padding_mode=padding, align_corners=False)[0]

    mode = zoom["interpolation"]
    padding = {"constant": "zeros", "reflect": "reflection", "replicate": "border"}[zoom["padding_mode"]]
    fill = zoom["fill_value"] if padding == "zeros" else 0.0
    radio = sample(tensor[0:1] - fill, mode, padding) + fill
    if tensor.shape[0] == 1:
        return radio
    chandra = tensor[1:2]
    valid = ((tensor[2:3] > .5) & torch.isfinite(chandra)).to(tensor.dtype)
    weighted = sample(torch.where(valid.bool(), chandra, 0), mode)
    weights = sample(valid, mode)
    coverage = sample(valid, "nearest") > .5
    coverage &= (grid[0].abs() <= 1).all(dim=-1)[None]
    coverage &= weights > 1e-6
    chandra = torch.where(coverage, weighted / weights.clamp_min(1e-6), 0)
    return torch.cat((radio, chandra, coverage.to(tensor.dtype)))
