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
        print(ax, metric, title)
        for split, label in (("train", "Entraînement"), ("validation", "Validation")):
            ax.plot([r["epoch"] for r in history], [r[split][metric] for r in history],
                    marker="o", label=label)
        ax.set(xlabel="Époque", ylabel=title)
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
