"""Bounded, opt-in figures; FITS/gallery work delegates to the existing renderer."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)

from .registry import epoch_table, prediction_table
from .comparison import aggregate_history, aggregate_metrics


def plot_history(runs, metric="loss", aggregate=False):
    epochs = epoch_table(runs)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    if epochs.empty:
        ax.set_title("No history available")
        return fig
    for prefix, style in [("train", "--"), ("validation", "-")]:
        column = f"{prefix}.{metric}"
        if column not in epochs:
            continue
        if aggregate:
            table = aggregate_history(epochs, column)
            for (config, protocol, phase), group in table.groupby(
                ["configuration_id", "protocol_id", "phase"]
            ):
                x = group.phase_epoch.to_numpy(float)
                mean = group["mean"].to_numpy(float)
                std = group["std"].to_numpy(float)
                run = next(r for r in runs if r.configuration_id == config)
                ax.plot(
                    x,
                    mean,
                    style,
                    label=f"{run.configuration_name} / {phase} / {prefix}",
                )
                ax.fill_between(x, mean - std, mean + std, alpha=0.15)
        else:
            for run in runs:
                group = epochs[epochs.run_id == run.run_id]
                label = run.run_name
                ax.plot(group.epoch, group[column], style, label=f"{label} / {prefix}")
                if prefix == "validation" and run.metrics.get("best_epoch") is not None:
                    ax.axvline(
                        run.metrics["best_epoch"],
                        color="gray",
                        alpha=0.3,
                        linestyle=":",
                    )
                if (
                    prefix == "validation"
                    and run.metrics.get("stopped_early")
                    and not group.empty
                ):
                    ax.scatter(
                        group.epoch.iloc[-1],
                        group[column].iloc[-1],
                        marker="x",
                        color="red",
                    )
                for _, phase in group.groupby("phase", sort=False):
                    if phase.epoch.min() != group.epoch.min():
                        ax.axvline(phase.epoch.min(), color="orange", alpha=0.4)
    ax.set(
        xlabel=(
            "Epoch within phase (available folds only)" if aggregate else "Global epoch"
        ),
        ylabel=metric,
    )
    if ax.lines:
        ax.legend(fontsize=6)
    return fig


def plot_performance(table, metric="macro_f1", kind="bar"):
    """Compare completed runs within each compatibility protocol.

    kind='heatmap': fold scores plus a Mean column, sorted by mean.
    kind='boxplot': horizontal boxes with individual, fold-colored observations.
    kind='strip' and kind='bar' preserve the original presentations.
    """
    if table.empty or metric not in table:
        fig, ax = plt.subplots()
        ax.set_title(f"{metric}: unavailable")
        return fig

    table = table.loc[(table.status == "completed") & table[metric].notna()].copy()
    if table.empty:
        fig, ax = plt.subplots()
        ax.set_title(f"{metric}: no completed runs")
        return fig

    # Compatibility is defined by the metric table, not by TrainingRun attributes.
    group_cols = ["protocol_id", "metric_source"]
    groups = list(table.groupby(group_cols, sort=True, dropna=False))
    max_configs = max(g.configuration_id.nunique() for _, g in groups)
    is_horizontal = kind in ("heatmap", "boxplot")
    height_per_group = max(3.5, 0.62 * max_configs + 1.6) if is_horizontal else 3.5
    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(13 if is_horizontal else 10, height_per_group * len(groups)),
        squeeze=False,
        layout="constrained",
    )

    from matplotlib.lines import Line2D

    for ax, ((protocol, source), group) in zip(axes.flat, groups):
        means = (
            group.groupby("configuration_id")[metric]
            .mean()
            .sort_values(ascending=False)
        )
        configs = (
            means.index.tolist()
            if is_horizontal
            else list(group.configuration_id.unique())
        )
        labels = [
            "\n".join(
                group.loc[group.configuration_id == c, "configuration_name"]
                .dropna()
                .astype(str)
                .unique()
            )
            for c in configs
        ]

        if kind == "heatmap":
            data = group.pivot_table(
                index="configuration_id", columns="fold", values=metric, aggfunc="mean"
            ).reindex(configs)
            if not data.empty and data.shape[1]:
                display_data = data.copy()
                display_data["Mean"] = data.mean(axis=1)
                values = display_data.to_numpy(dtype=float)
                # Scores such as F1, AUC, and AP have a natural [0, 1] range.
                bounded = metric.lower() in {
                    "macro_f1",
                    "f1",
                    "accuracy",
                    "roc_auc",
                    "auc",
                    "ap",
                    "average_precision",
                    "precision",
                    "recall",
                }
                finite = values[np.isfinite(values)]
                vmin, vmax = (
                    (0.0, 1.0)
                    if bounded
                    else (
                        (float(finite.min()), float(finite.max()))
                        if finite.size
                        else (0.0, 1.0)
                    )
                )
                if vmin == vmax:
                    vmax = vmin + 1.0
                heatmap = ax.imshow(
                    values, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax
                )
                ax.set_yticks(np.arange(len(configs)), labels)
                ax.set_xticks(
                    np.arange(len(display_data.columns)),
                    [str(c) for c in display_data.columns],
                )
                ax.axvline(len(data.columns) - 0.5, color="black", linewidth=1.8)
                for i, row in enumerate(values):
                    for j, value in enumerate(row):
                        if np.isfinite(value):
                            rgba = heatmap.cmap(heatmap.norm(value))
                            luminance = (
                                0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                            )
                            ax.text(
                                j,
                                i,
                                f"{value:.3f}",
                                ha="center",
                                va="center",
                                color="black" if luminance > 0.52 else "white",
                                fontsize=9,
                                fontweight=(
                                    "bold" if j == len(data.columns) else "normal"
                                ),
                            )
                ax.set_xlabel("Fold / mean across folds")
                fig.colorbar(heatmap, ax=ax, label=metric, shrink=0.85)

        elif kind == "boxplot":
            positions = np.arange(len(configs))
            box_data = [
                group.loc[group.configuration_id == c, metric].to_numpy(dtype=float)
                for c in configs
            ]
            ax.boxplot(
                box_data,
                positions=positions,
                vert=False,
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.5},
                boxprops={"facecolor": "#dce8f2", "edgecolor": "#54718b"},
                whiskerprops={"color": "#54718b"},
                capprops={"color": "#54718b"},
            )
            folds = sorted(group["fold"].dropna().unique())
            cmap = plt.get_cmap("tab10")
            fold_colors = {fold: cmap(i % 10) for i, fold in enumerate(folds)}
            for i, config in enumerate(configs):
                config_data = group[group.configuration_id == config]
                for _, record in config_data.iterrows():
                    ax.scatter(
                        record[metric],
                        i,
                        color=fold_colors.get(record["fold"], "gray"),
                        s=48,
                        edgecolor="white",
                        linewidth=0.8,
                        zorder=3,
                    )
                ax.scatter(
                    means.loc[config], i, marker="D", s=65, color="black", zorder=4
                )
            ax.set_yticks(positions, labels)
            ax.invert_yaxis()
            ax.set_xlabel(metric)
            if metric.lower() in {
                "macro_f1",
                "f1",
                "accuracy",
                "roc_auc",
                "auc",
                "ap",
                "average_precision",
                "precision",
                "recall",
                "brier",
                "brier_score",
            }:
                ax.set_xlim(0, 1)
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    color="white",
                    markerfacecolor=fold_colors[fold],
                    markeredgecolor="white",
                    markersize=8,
                    label=f"Fold {fold}",
                )
                for fold in folds
            ]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    linestyle="None",
                    color="black",
                    markersize=7,
                    label="Mean",
                )
            )
            ax.legend(handles=handles, loc="lower right", fontsize=8, ncol=2)

        elif kind == "strip":
            for i, config in enumerate(configs):
                values = group.loc[group.configuration_id == config, metric]
                ax.scatter(np.full(len(values), i), values)
            ax.set_xticks(range(len(configs)), labels)
        elif kind == "bar":
            summary = (
                aggregate_metrics(group, [metric])
                .set_index("configuration_id")
                .reindex(configs)
            )
            ax.bar(
                range(len(configs)),
                summary[metric + "_mean"],
                yerr=summary[metric + "_std"].fillna(0),
            )
            ax.set_xticks(
                range(len(configs)),
                [
                    f'{label}\nn_folds={int(summary.loc[c, "n_folds"])}'
                    for c, label in zip(configs, labels)
                ],
            )
        else:
            raise ValueError(
                f"Unknown kind: {kind!r}. Expected heatmap, boxplot, strip, or bar"
            )

        protocol_text = str(protocol)
        ax.set_title(
            f"{group.iloc[0].task or group.iloc[0].task_type} / "
            f"{group.iloc[0].split} / protocol {protocol_text[:8]} — {metric}"
        )
    return fig


def _plot_diagnostic_history(run, axes):
    """One shared history row, including for runs without detailed predictions."""
    history = epoch_table([run])
    for ax, metric, title in zip(
        axes, ("loss", "macro_f1", "accuracy"), ("Loss", "Macro-F1", "Accuracy")
    ):
        available = False
        for split, label, style in [
            ("train", "Training", "--"),
            ("validation", "Validation", "-"),
        ]:
            column = f"{split}.{metric}"
            if column not in history or not history[column].notna().any():
                continue
            ax.plot(history.epoch, history[column], style, label=label)
            available = True
        if available:
            best = run.metrics.get("best_epoch")
            if best is not None:
                ax.axvline(
                    best,
                    color="gray",
                    linestyle=":",
                    alpha=0.7,
                    label="Best epoch",
                )
            for _, phase in list(history.groupby("phase", sort=False))[1:]:
                ax.axvline(
                    phase.epoch.iloc[0], color="orange", alpha=0.5, linestyle=":"
                )
            ax.legend(fontsize=8)
        else:
            ax.text(
                0.5, 0.5, "History unavailable", ha="center", transform=ax.transAxes
            )
        ax.set(title=f"{title} per epoch", xlabel="Epoch", ylabel=title)
        ax.grid(alpha=0.2)


def plot_diagnostics(run, split="test", cohort="all"):
    table = prediction_table([run], split, cohort)
    labels = run.diagnostic_labels
    diagnostic_rows = max(1, len(labels))
    fig = plt.figure(figsize=(16, 3.5 * (diagnostic_rows + 1)), layout="constrained")
    grid = fig.add_gridspec(diagnostic_rows + 1, 12)
    axes = [
        [fig.add_subplot(grid[row, col * 3 : (col + 1) * 3]) for col in range(4)]
        for row in range(diagnostic_rows)
    ]
    history_axes = [
        fig.add_subplot(grid[-1, col * 4 : (col + 1) * 4]) for col in range(3)
    ]
    _plot_diagnostic_history(run, history_axes)
    for axes_row, label in zip(axes, labels):
        if table.empty:
            for ax in axes_row:
                ax.text(
                    0.5,
                    0.5,
                    "Detailed predictions unavailable",
                    ha="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
            continue
        frame = table[(table.label == label) & table.known]
        y, score = frame.truth.to_numpy(), frame.score.to_numpy()
        if len(frame) and frame.prediction.notna().all():
            matrix = confusion_matrix(y, frame.prediction, labels=[0, 1])
            support = matrix.sum(axis=1, keepdims=True)
            percentages = np.divide(
                100.0 * matrix,
                support,
                out=np.zeros_like(matrix, dtype=float),
                where=support != 0,
            )
            axes_row[0].imshow(percentages, cmap="Blues", vmin=0, vmax=100)
            for i in range(2):
                for j in range(2):
                    text = (
                        f"{percentages[i, j]:.1f}% ({matrix[i, j]})"
                        if support[i, 0]
                        else "— (0)"
                    )
                    axes_row[0].text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color="white" if percentages[i, j] > 50 else "black",
                    )
        else:
            axes_row[0].text(
                0.5,
                0.5,
                "Thresholds or known labels unavailable",
                ha="center",
                wrap=True,
            )
        negative = (
            next((name for name in run.labels if name != label), f"not_{label}")
            if run.task_type == "binary"
            else f"not_{label}"
        )
        class_names = [negative, label]
        axes_row[0].set(
            title=f"{label}: Confusion matrix (% by true class)",
            xlabel="Prediction",
            ylabel="True class",
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=class_names,
            yticklabels=class_names,
        )
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, score)
            precision, recall, _ = precision_recall_curve(y, score)

            roc_auc = roc_auc_score(y, score)
            ap = average_precision_score(y, score)

            axes_row[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            axes_row[1].plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
            axes_row[1].legend()

            axes_row[2].step(recall, precision, where="post", label=f"AP = {ap:.3f}")
            axes_row[2].axhline(
                np.mean(y),
                linestyle=":",
                color="gray",
                label=f"Baseline = {np.mean(y):.3f}",
            )
            axes_row[2].legend()
        axes_row[1].set(title="ROC", xlabel="FPR", ylabel="TPR")
        axes_row[2].set(title="Precision–Recall", xlabel="Recall", ylabel="Precision")
        # Calibration replaces the probability histogram in the fourth panel.
        ax_cal = axes_row[3]
        ax_cal.plot(
            [0, 1], [0, 1], "--", color="gray", alpha=0.6, label="Perfect calibration"
        )
        valid = np.isfinite(score) & np.isfinite(y) & (score >= 0) & (score <= 1)
        if valid.any():
            y_valid = y[valid].astype(int)
            score_valid = score[valid].astype(float)
            prob_true, prob_pred = calibration_curve(
                y_valid, score_valid, n_bins=5, strategy="quantile"
            )
            brier = brier_score_loss(y_valid, score_valid)
            ax_cal.plot(
                prob_pred, prob_true, "o-", label=f"Model (Brier = {brier:.3f})"
            )
        ax_cal.set(
            title="Calibration",
            xlabel="Mean predicted probability",
            ylabel="Observed positive fraction",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax_cal.legend(fontsize=7)
    fig.suptitle(f"{run.run_name} / {split}")
    return fig


def plot_galleries(
    run,
    *,
    processed_root,
    split="test",
    cohort="all",
    max_clusters=12,
    radio_template="{cluster_id}/{cluster_id}_RAW_processed_size128_v2.fits",
    chandra_template="{cluster_id}/{cluster_id}_CHANDRA_processed_size128_v2.fits",
    chandra_only=False,
    view="overlay",
):
    from ..multilabel_analysis import plot_classification_gallery

    if view not in ("overlay", "radio", "chandra"):
        raise ValueError("Gallery view: overlay, radio or chandra")
    frame = prediction_table([run], split, cohort)
    figures = {}
    if frame.empty:
        return figures
    if run.task_type == "multilabel":
        groups = [
            (
                f"{label}_{outcome}",
                frame[(frame.label == label) & (frame.correct == correct)],
            )
            for label in run.labels
            for outcome, correct in [("correct", True), ("incorrect", False)]
        ]
    else:
        label = run.diagnostic_labels[0]
        data = frame[frame.label == label]
        groups = [
            (name, data[(data.truth == y) & (data.prediction == p)])
            for name, y, p in [("TP", 1, 1), ("TN", 0, 0), ("FP", 0, 1), ("FN", 1, 0)]
        ]
    for name, data in groups:
        entries = []
        for row in data.head(max_clusters).to_dict("records"):
            # Existing renderer understands either binary or RR/RH title fields.
            row.update(
                positive_class=row["label"],
                probability=row["score"],
                label=int(row["truth"]),
            )
            if view == "radio":
                row["has_chandra"] = False
            entries.append(row)
        figures[name] = plot_classification_gallery(
            entries,
            title=f"{run.run_name} / {name}",
            processed_root=processed_root,
            radio_template=radio_template,
            chandra_template=chandra_template,
            chandra_only=chandra_only or view == "chandra",
        )
    return figures


def plot_paired(table):
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    if not table.empty:
        paired = table[table.paired]
        ax.scatter(range(len(paired)), paired.delta)
        ax.set_xticks(
            range(len(paired)),
            [f"{r.left_name} → {r.right_name}" for r in paired.itertuples()],
            rotation=45,
        )
    ax.axhline(0, color="gray")
    ax.set(ylabel="Paired metric difference (right − left)")
    return fig


def plot_oof_diagnostics(runs, split="test", cohort="all", n_bins=5):
    """Four OOF panels per label; fold-wise mean ± sample SD for all metrics.

    Requires one test prediction per cluster and label, per configuration/protocol.
    All runs must represent distinct folds of the SAME CV experiment.
    """
    if not runs:
        raise ValueError("Provide at least one run")
    config_ids = {r.configuration_id for r in runs}
    if len(config_ids) != 1:
        raise ValueError("Runs must share the same configuration_id")
    folds = [r.fold for r in runs]
    if len(set(folds)) != len(folds):
        raise ValueError("One run per fold is required")

    table = prediction_table(runs, split, cohort)
    if table.empty:
        raise ValueError("No detailed predictions available")
    if "cluster_id" not in table:
        raise ValueError("cluster_id is required to verify unique OOF predictions")
    if "run_id" not in table:
        raise ValueError("run_id is required to associate predictions with folds")

    labels = runs[0].diagnostic_labels
    fig, axes = plt.subplots(
        len(labels),
        4,
        figsize=(17, 4 * len(labels)),
        squeeze=False,
        layout="constrained",
    )
    fold_stats = []

    for row_index, label in enumerate(labels):
        ax_cm, ax_roc, ax_pr, ax_cal = axes[row_index]
        frame = table[(table.label == label) & table.known].copy()
        if frame.empty:
            for ax in axes[row_index]:
                ax.set_axis_off()
            continue
        if frame.cluster_id.isna().any() or frame.cluster_id.duplicated().any():
            raise ValueError(
                f"Repeated or missing cluster IDs for {label}: not valid OOF"
            )
        if frame[["truth", "score"]].isna().any().any():
            raise ValueError(f"Missing truth or scores for {label}")
        if not np.isfinite(frame.score.to_numpy(float)).all():
            raise ValueError(f"Non-finite scores for {label}")
        if not frame.score.between(0, 1).all():
            raise ValueError(f"Scores must be probabilities in [0,1] for {label}")
        if not frame.truth.isin([0, 1]).all():
            raise ValueError(f"Truth must be binary for {label}")

        matrices = []
        for run in runs:
            part = frame[frame.run_id == run.run_id]
            if part.empty:
                continue
            y_fold = part.truth.to_numpy(int)
            s_fold = part.score.to_numpy(float)
            stats = {
                "label": label,
                "fold": run.fold,
                "brier": brier_score_loss(y_fold, s_fold),
                "auc": np.nan,
                "ap": np.nan,
            }
            if len(np.unique(y_fold)) == 2:
                stats["auc"] = roc_auc_score(y_fold, s_fold)
                stats["ap"] = average_precision_score(y_fold, s_fold)
            fold_stats.append(stats)
            if part.prediction.notna().all():
                cm = confusion_matrix(
                    y_fold, part.prediction.to_numpy(int), labels=[0, 1]
                )
                support = cm.sum(axis=1)
                percentages = np.full((2, 2), np.nan)
                np.divide(
                    cm * 100.0,
                    support[:, None],
                    out=percentages,
                    where=support[:, None] > 0,
                )
                matrices.append(percentages)

        if matrices:
            stack = np.stack(matrices)
            counts = np.isfinite(stack).sum(axis=0)
            avg = np.divide(
                np.nansum(stack, axis=0),
                counts,
                out=np.full((2, 2), np.nan),
                where=counts > 0,
            )
            sd = np.full((2, 2), np.nan)
            for i in range(2):
                for j in range(2):
                    vals = stack[:, i, j]
                    vals = vals[np.isfinite(vals)]
                    if len(vals) >= 2:
                        sd[i, j] = np.std(vals, ddof=1)
            ax_cm.imshow(avg, cmap="Blues", vmin=0, vmax=100)
            for i in range(2):
                for j in range(2):
                    txt = (
                        f"{avg[i,j]:.1f}% ± {sd[i,j]:.1f}%"
                        if np.isfinite(sd[i, j])
                        else (
                            f"{avg[i,j]:.1f}% (SD N/A)"
                            if np.isfinite(avg[i, j])
                            else "N/A"
                        )
                    )
                    ax_cm.text(
                        j,
                        i,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=(
                            "white"
                            if np.isfinite(avg[i, j]) and avg[i, j] > 60
                            else "black"
                        ),
                    )
        negative = (
            next((name for name in runs[0].labels if name != label), f"not_{label}")
            if runs[0].task_type == "binary"
            else f"not_{label}"
        )
        ax_cm.set(
            title=f"{label}: Mean confusion matrix (fold % ± SD)",
            xlabel="Prediction",
            ylabel="True class",
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=[negative, label],
            yticklabels=[negative, label],
        )

        y = frame.truth.to_numpy(int)
        score = frame.score.to_numpy(float)
        stats_df = pd.DataFrame([v for v in fold_stats if v["label"] == label])

        def summary(metric):
            values = stats_df[metric].dropna().to_numpy(float)
            if not len(values):
                return "N/A"
            if len(values) == 1:
                return f"{values[0]:.3f} (SD N/A)"
            return f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"

        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, score)
            precision, recall, _ = precision_recall_curve(y, score)
            ax_roc.plot(fpr, tpr, label=f"Fold AUC: {summary('auc')}")
            ax_pr.step(
                recall, precision, where="post", label=f"Fold AP: {summary('ap')}"
            )
            ax_pr.axhline(
                y.mean(),
                linestyle=":",
                color="gray",
                label=f"Prevalence = {y.mean():.3f}",
            )
        ax_roc.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
        ax_roc.set(
            title="OOF ROC", xlabel="FPR", ylabel="TPR", xlim=(0, 1), ylim=(0, 1)
        )
        ax_pr.set(
            title="OOF Precision–Recall",
            xlabel="Recall",
            ylabel="Precision",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax_roc.legend(fontsize=8)
        ax_pr.legend(fontsize=8)

        prob_true, prob_pred = calibration_curve(
            y, score, n_bins=n_bins, strategy="quantile"
        )
        ax_cal.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax_cal.plot(prob_pred, prob_true, "o-", label=f"Fold Brier: {summary('brier')}")
        ax_cal.set(
            title="OOF Calibration",
            xlabel="Mean predicted probability",
            ylabel="Observed positive fraction",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax_cal.legend(fontsize=8)

    fig.suptitle(
        f"{runs[0].configuration_name} / {split} / {len(runs)} folds — OOF summary"
    )
    return fig, pd.DataFrame(fold_stats)
