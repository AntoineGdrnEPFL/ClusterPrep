"""Figures retournées à l'appelant ; aucun changement de backend global."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from reproject import reproject_interp


def plot_history(run_dir, output_path=None):
    """Courbes train/validation d'une expérience enregistrée, sans scores test."""
    import json
    run_dir = Path(run_dir)
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    if not history:
        raise ValueError("Historique vide")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)
    for ax, metric, title in zip(axes, ("loss", "macro_f1", "accuracy","roc_auc_macro"), ("Entropie croisée", "F1 macro", "Exactitude", "AUC ROC")):
        for split, label in (("train", "Entraînement"), ("validation", "Validation")):
            if not any(r.get(split) is not None for r in history):
                continue
            ax.plot([r["epoch"] for r in history], [(r.get(split) or {}).get(metric) for r in history],
                    marker="o", label=label)
        ax.set(xlabel="Époque", ylabel=title)
        previous = None
        for record in history:
            phase = record.get("phase")
            if phase is not None and phase != previous:
                ax.axvline(record["epoch"] - .5, color="grey", linestyle="--", alpha=.5)
                ax.text(record["epoch"], .98, phase, rotation=90, va="top",
                        transform=ax.get_xaxis_transform(), fontsize=8)
                previous = phase
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle(run_dir.name)
    return _save(fig, output_path)


def _norm(image):
    finite = np.asarray(image)[np.isfinite(image)]
    lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0., 1.)
    if hi <= lo:
        hi = lo + max(abs(lo) * 1e-6, 1e-8)
    return AsinhNorm(linear_width=(hi - lo) / 10, vmin=lo, vmax=hi)


def _save(fig, output_path):
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
    return fig


def plot_sample(sample, output_path=None):
    """Affiche un échantillon du Dataset, dont le masque de couverture Chandra."""
    channels = [name for name in ("raw", "chandra", "chandra_mask") if name in sample]
    fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 4),
                             squeeze=False, constrained_layout=True)
    for ax, name in zip(axes[0], channels):
        value = sample[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        image = np.asarray(value).squeeze()
        artist = ax.imshow(image, origin="lower", vmin=0, vmax=1,
                           cmap="viridis" if name == "raw" else "inferno")
        ax.set_title({"raw": "Radio normalisé", "chandra": "Chandra normalisé",
                      "chandra_mask": "Couverture Chandra"}[name])
        ax.set_xlabel("Pixel x")
        ax.set_ylabel("Pixel y")
        fig.colorbar(artist, ax=ax, shrink=.8)
    fig.suptitle(str(sample.get("cluster_id", "")))
    return _save(fig, output_path)


def plot_zoom_variants(dataset, source_index=0, factors=(.9, 1., 1.1, 1.2),
                       geometry_index=0, output_path=None, zoom=None):
    """Compare zooms of one clean source with shared colour scales per row.

    Reuses the dataset's preprocessing and a single normalized source. Noise
    and scattering are disabled in this diagnostic; no dataset is mutated.
    ``geometry_index`` uses the historical 0..23 rotation/flip ordering.
    """
    import torch
    from .augmentation import (GEOMETRIC_VARIANTS, decode_variant,
                               transform_aligned, validate_zoom)
    from .dataset import ClusterDataset

    if not 0 <= geometry_index < GEOMETRIC_VARIANTS:
        raise ValueError("geometry_index doit être entre 0 et 23")
    if not 0 <= source_index < len(dataset.keys):
        raise IndexError(source_index)
    config = validate_zoom({**dataset.zoom, **(zoom or {}),
                            "enabled": True, "factors": list(factors)})
    source = ClusterDataset(
        dataset.data_info, [dataset.keys[source_index]], list(dataset.class_to_idx),
        mode=dataset.mode, size=dataset.size, processed_dir=dataset.processed_dir,
        include_targets=False,
    )[0]
    names = [name for name in ("raw", "chandra", "chandra_mask") if name in source]
    tensor = torch.cat([source[name] for name in names])
    angle, flip, _ = decode_variant(geometry_index)
    variants = [transform_aligned(tensor, angle, flip, scale, config).numpy()
                for scale in config["factors"]]
    fig, axes = plt.subplots(len(names), len(variants), squeeze=False,
                             figsize=(4 * len(variants), 3 * len(names)),
                             constrained_layout=True)
    labels = {"raw": "Radio", "chandra": "CHANDRA", "chandra_mask": "Mask"}
    for row, name in enumerate(names):
        lo = min(0., min(float(v[row].min()) for v in variants))
        hi = max(1., max(float(v[row].max()) for v in variants))
        for col, (scale, value) in enumerate(zip(config["factors"], variants)):
            ax = axes[row, col]
            artist = ax.imshow(value[row], origin="lower", vmin=lo, vmax=hi,
                               cmap="gray" if name == "chandra_mask" else
                               "viridis" if name == "raw" else "inferno")
            ax.set_title(f"{labels[name]} — {scale:g}×")
        fig.colorbar(artist, ax=list(axes[row]), shrink=.8)
    fig.suptitle(f"{source['cluster_id']} — rotation {angle}°, flip {flip}")
    return _save(fig, output_path)


def plot_comparison(raw_image, chandra_image, raw_wcs, chandra_wcs,
                    raw_title="Radio", chandra_title="Chandra", output_path=None):
    """Compare les images natives et Chandra reprojeté sur la grille radio."""
    aligned, _ = reproject_interp((chandra_image, chandra_wcs), raw_wcs,
                                  shape_out=raw_image.shape)
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    for i, (data, wcs, title) in enumerate(((raw_image, raw_wcs, raw_title),
            (chandra_image, chandra_wcs, chandra_title),
            (aligned, raw_wcs, "Chandra sur la grille radio")), 1):
        ax = fig.add_subplot(1, 3, i, projection=wcs)
        artist = ax.imshow(data, origin="lower", norm=_norm(data),
                           cmap="viridis" if i == 1 else "inferno")
        ax.set_title(title)
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
        ax.coords.grid(color="white", alpha=.25, ls=":")
        fig.colorbar(artist, ax=ax, shrink=.75)
    return _save(fig, output_path)
