"""Métriques multiclasses sérialisables sans NaN ; ordre des classes explicite."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, roc_auc_score, average_precision_score)


def classification_metrics(labels, probabilities, class_names):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    n_classes = len(class_names)
    if not labels.size or probabilities.shape != (len(labels), n_classes):
        raise ValueError("Échantillons non vides et probabilités [N, C] attendus")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Probabilités invalides")
    if (labels < 0).any() or (labels >= n_classes).any():
        raise ValueError("Labels hors de l'ordre des classes")
    predicted = probabilities.argmax(axis=1)
    indices = list(range(n_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=indices, zero_division=0)
    aucs, aps, per_class = [], [], {}
    for i, name in enumerate(class_names):
        binary = labels == i
        # Une seule classe présente : pas d'AUC/AP interprétable.
        auc = float(roc_auc_score(binary, probabilities[:, i])) if 0 < binary.sum() < len(labels) else None
        ap = float(average_precision_score(binary, probabilities[:, i])) if auc is not None else None
        aucs.append(auc)
        aps.append(ap)
        per_class[name] = {"precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i]), "roc_auc": auc,
            "average_precision": ap}
    return {"n_samples": len(labels), "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(f1_score(labels, predicted, labels=indices, average="weighted", zero_division=0)),
        "roc_auc_macro": float(np.mean(aucs)) if all(a is not None for a in aucs) else None,
        "average_precision_macro": float(np.mean(aps)) if all(a is not None for a in aps) else None,
        "confusion_matrix": confusion_matrix(labels, predicted, labels=indices).tolist(),
        "per_class": per_class}
