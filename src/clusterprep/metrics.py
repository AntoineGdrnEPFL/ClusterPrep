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


def resolve_thresholds(thresholds=None, label_names=("RR", "RH"), threshold=0.5):
    """Return thresholds in label order; legacy scalar threshold remains supported."""
    names = tuple(label_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Noms de labels non vides et uniques requis")
    if thresholds is not None and threshold != 0.5:
        raise ValueError("Utiliser threshold ou thresholds, pas les deux")
    if thresholds is None:
        values = [threshold] * len(names)
    elif isinstance(thresholds, dict):
        if set(thresholds) != set(names):
            raise ValueError(f"Seuils requis pour {names}")
        values = [thresholds[name] for name in names]
    else:
        values = thresholds
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Seuils numériques requis") from error
    if array.shape != (len(names),) or not np.isfinite(array).all() or ((array <= 0) | (array >= 1)).any():
        raise ValueError("Un seuil fini strictement entre 0 et 1 par label est requis")
    return dict(zip(names, map(float, array)))


def _multilabel_arrays(targets, probabilities, label_names, mask=None, target_mask=None):
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=float)
    if (targets.ndim != 2 or targets.shape[1] != len(label_names)
            or not len(targets) or probabilities.shape != targets.shape):
        raise ValueError("Targets/probabilités non vides [N, nombre_labels] attendus")
    if mask is not None and target_mask is not None:
        raise ValueError("Utiliser mask ou target_mask, pas les deux")
    mask = target_mask if target_mask is not None else mask
    observed = np.ones_like(targets, dtype=bool) if mask is None else np.asarray(mask)
    if observed.shape != targets.shape or not np.isin(observed, [0, 1]).all():
        raise ValueError("Masque de labels binaire de forme [N, C] requis")
    observed = observed.astype(bool)
    if not np.isin(targets[observed], [0, 1]).all():
        raise ValueError("Targets connus binaires 0/1 attendus")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any()):
        raise ValueError("Probabilités invalides")
    # Unknown values may be NaN/-1; they are never counted as negatives.
    return np.where(observed, targets, 0).astype(int), probabilities, observed


def multilabel_metrics(targets, probabilities, label_names=("RR", "RH"), threshold=0.5,
                       mask=None, *, thresholds=None, target_mask=None):
    """Scores sur les annotations connues, avec seuils indépendants par label.

    AP et ROC-AUC exigent deux classes connues (sinon None). Les moyennes macro
    ignorent les scores non définis et indiquent leur nombre de labels évaluables.
    Hamming utilise les cellules connues ; subset utilise les lignes complètes.
    Les cibles masquées peuvent être 0, 1, -1 ou NaN sans changer les scores.
    """
    cutoffs = resolve_thresholds(thresholds, label_names, threshold)
    targets, probabilities, observed = _multilabel_arrays(
        targets, probabilities, label_names, mask, target_mask)
    predicted = probabilities >= np.array(list(cutoffs.values()))
    per_label = {}
    for j, name in enumerate(label_names):
        known = observed[:, j]
        truth, pred, proba = targets[known, j], predicted[known, j], probabilities[known, j]
        precision = recall = f1 = auc = ap = balanced = brier = None
        matrix = [[0, 0], [0, 0]]
        if known.any():
            precision, recall, f1, _ = precision_recall_fscore_support(
                truth, pred, average="binary", zero_division=0)
            precision, recall, f1 = map(float, (precision, recall, f1))
            matrix = confusion_matrix(truth, pred, labels=[0, 1]).tolist()
            brier = float(np.mean((proba - truth) ** 2))
        if len(np.unique(truth)) == 2:
            auc = float(roc_auc_score(truth, proba))
            ap = float(average_precision_score(truth, proba))
            balanced = float((np.mean(pred[truth == 1]) + np.mean(~pred[truth == 0])) / 2)
        tn, fp, fn, tp = np.asarray(matrix).ravel().tolist()
        per_label[name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "roc_auc": auc, "average_precision": ap, "balanced_accuracy": balanced,
            "brier": brier, "threshold": cutoffs[name],
            "support": int(truth.sum()), "n_positive": int(truth.sum()),
            "n_negative": int(len(truth) - truth.sum()), "n_known": int(known.sum()),
            "prevalence": float(truth.mean()) if known.any() else None,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "confusion_matrix": matrix,
        }

    aggregates = {}
    for output, source in (("macro_f1", "f1"), ("roc_auc_macro", "roc_auc"),
                           ("average_precision_macro", "average_precision"),
                           ("balanced_accuracy", "balanced_accuracy"), ("brier_macro", "brier")):
        values = [v[source] for v in per_label.values() if v[source] is not None]
        aggregates[output] = float(np.mean(values)) if values else None
        aggregates[f"{output}_n_labels"] = len(values)
    fully_known = observed.all(1)
    subset = (float(np.mean(np.all(targets[fully_known] == predicted[fully_known], axis=1)))
              if fully_known.any() else None)
    hamming = float(np.mean(targets[observed] == predicted[observed])) if observed.any() else None
    support = sum(v["support"] for v in per_label.values())
    weighted = (sum((v["f1"] or 0) * v["support"] for v in per_label.values()) / support
                if support else (0.0 if observed.any() else None))
    return {
        "n_samples": len(targets), "n_fully_known": int(fully_known.sum()),
        "n_known": int(observed.sum()), "thresholds": cutoffs,
        "accuracy": subset, "subset_accuracy": subset,
        "hamming_accuracy": hamming, "hamming_loss": 1 - hamming if hamming is not None else None,
        "micro_f1": float(f1_score(targets[observed], predicted[observed], zero_division=0)) if observed.any() else None,
        "weighted_f1": weighted, **aggregates, "per_label": per_label,
    }


def select_multilabel_thresholds(validation_targets, validation_probabilities, *,
                                target_mask=None, label_names=("RR", "RH"),
                                method="f1", grid=None, min_precision=0.8,
                                fallback=None, split="validation"):
    """Sélection déterministe sur validation exclusivement, jamais sur le test.

    Grille par défaut : 0.05, 0.10, ..., 0.95. Ex æquo : plus proche de 0.5,
    puis seuil le plus élevé. Sans deux classes ou contrainte réalisable, fallback.
    """
    if split != "validation":
        raise ValueError("Les seuils doivent être sélectionnés sur validation uniquement")
    if method not in {"fixed", "f1", "recall_at_precision"}:
        raise ValueError("Méthode attendue : fixed, f1 ou recall_at_precision")
    if not np.isfinite(min_precision) or not 0 <= min_precision <= 1:
        raise ValueError("min_precision doit être entre 0 et 1")
    cutoffs = resolve_thresholds(fallback, label_names)
    y, p, known = _multilabel_arrays(validation_targets, validation_probabilities, label_names, target_mask=target_mask)
    grid = np.asarray(list(grid) if grid is not None else np.arange(5, 100, 5) / 100, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or ((grid <= 0) | (grid >= 1)).any():
        raise ValueError("Grille non vide de seuils finis entre 0 et 1 requise")
    grid = np.unique(grid)
    base = multilabel_metrics(y, p, label_names, target_mask=known, thresholds=cutoffs)
    scans = [] if method == "fixed" else [multilabel_metrics(y, p, label_names, target_mask=known, threshold=float(t)) for t in grid]
    details = {}
    for name in label_names:
        scores = base["per_label"][name]
        detail = {k: scores[k] for k in ("n_known", "n_positive", "n_negative")}
        reason = "fixed" if method == "fixed" else "single_class_or_no_known_labels"
        if method != "fixed" and scores["n_positive"] and scores["n_negative"]:
            candidates = [(float(t), result["per_label"][name]) for t, result in zip(grid, scans)]
            if method == "recall_at_precision":
                candidates = [(t, m) for t, m in candidates if m["precision"] >= min_precision]
            if candidates:
                key = "f1" if method == "f1" else "recall"
                best, _ = max(candidates, key=lambda pair: (pair[1][key], -round(abs(pair[0] - .5), 12), pair[0]))
                cutoffs[name] = best
                reason = "selected"
            else:
                reason = "precision_constraint_unattainable"
        details[name] = {**detail, "status": reason, "threshold": cutoffs[name]}
    return {"method": method, "selection_split": "validation" if method != "fixed" else None,
            "thresholds": cutoffs, "grid": grid.tolist() if method != "fixed" else [],
            "min_precision": float(min_precision) if method == "recall_at_precision" else None,
            "tie_break": "closest_to_0.5_then_largest", "per_label": details}
