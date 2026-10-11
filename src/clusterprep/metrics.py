"""Métriques multiclasses sérialisables sans NaN ; ordre des classes explicite."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


def classification_metrics(labels, probabilities, class_names):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    n_classes = len(class_names)
    if not labels.size or probabilities.shape != (len(labels), n_classes):
        raise ValueError("Échantillons non vides et probabilités [N, C] attendus")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Probabilités invalides")
    if (labels < 0).any() or (labels >= n_classes).any():
        raise ValueError("Labels hors de l'ordre des classes")
    predicted = probabilities.argmax(axis=1)
    indices = list(range(n_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=indices, zero_division=0
    )
    aucs, aps, per_class = [], [], {}
    for i, name in enumerate(class_names):
        binary = labels == i
        # Une seule classe présente : pas d'AUC/AP interprétable.
        auc = (
            float(roc_auc_score(binary, probabilities[:, i]))
            if 0 < binary.sum() < len(labels)
            else None
        )
        ap = (
            float(average_precision_score(binary, probabilities[:, i]))
            if auc is not None
            else None
        )
        aucs.append(auc)
        aps.append(ap)
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "roc_auc": auc,
            "average_precision": ap,
        }
    return {
        "n_samples": len(labels),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(
            f1_score(
                labels, predicted, labels=indices, average="weighted", zero_division=0
            )
        ),
        "roc_auc_macro": (
            float(np.mean(aucs)) if all(a is not None for a in aucs) else None
        ),
        "average_precision_macro": (
            float(np.mean(aps)) if all(a is not None for a in aps) else None
        ),
        "confusion_matrix": confusion_matrix(
            labels, predicted, labels=indices
        ).tolist(),
        "per_class": per_class,
    }


def multilabel_metrics(
    targets,
    probabilities,
    label_names=("RR", "RH"),
    threshold=0.5,
):
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    n_labels = len(label_names)

    if targets.ndim != 2 or targets.shape[1] != n_labels:
        raise ValueError("Targets attendus : [N, nombre_labels]")

    if probabilities.shape != targets.shape:
        raise ValueError("Probabilités et targets incompatibles")

    if len(targets) == 0:
        raise ValueError("Dataset vide")

    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Targets binaires 0/1 attendus")

    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Probabilités invalides")

    if not 0 < threshold < 1:
        raise ValueError("threshold doit être entre 0 et 1")

    predicted = (probabilities >= threshold).astype(int)

    per_label = {}
    aucs = []
    aps = []

    for index, name in enumerate(label_names):
        y_true = targets[:, index]
        y_pred = predicted[:, index]
        y_prob = probabilities[:, index]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        auc = (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2
            else None
        )

        ap = (
            float(average_precision_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2
            else None
        )

        if auc is not None:
            aucs.append(auc)

        if ap is not None:
            aps.append(ap)

        per_label[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": auc,
            "average_precision": ap,
            "support": int(y_true.sum()),
            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1],
            ).tolist(),
        }

    return {
        "n_samples": len(targets),
        # Exact match : toutes les étiquettes doivent être correctes
        "accuracy": float(accuracy_score(targets, predicted)),
        "subset_accuracy": float(accuracy_score(targets, predicted)),
        "hamming_accuracy": float(np.mean(targets == predicted)),
        "macro_f1": float(
            f1_score(
                targets,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "micro_f1": float(
            f1_score(
                targets,
                predicted,
                average="micro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                targets,
                predicted,
                average="weighted",
                zero_division=0,
            )
        ),
        "balanced_accuracy": (
            float(
                np.mean(
                    [
                        (
                            (
                                np.mean(predicted[targets[:, j] == 1, j] == 1)
                                + np.mean(predicted[targets[:, j] == 0, j] == 0)
                            )
                            / 2
                        )
                        for j in range(n_labels)
                        if np.any(targets[:, j] == 1) and np.any(targets[:, j] == 0)
                    ]
                )
            )
            if any(
                np.any(targets[:, j] == 1) and np.any(targets[:, j] == 0)
                for j in range(n_labels)
            )
            else None
        ),
        "roc_auc_macro": (float(np.mean(aucs)) if aucs else None),
        "average_precision_macro": (float(np.mean(aps)) if aps else None),
        "per_label": per_label,
    }
