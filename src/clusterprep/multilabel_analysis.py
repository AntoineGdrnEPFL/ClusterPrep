"""Analyse RR/RH d'artefacts sauvegardés, sans checkpoint, FITS ni réseau.

Les annotations des CSV sont la référence évaluée. Le manifeste permet de
vérifier identités, fuite, politique des candidats et présence des modalités.
Pandas/Jupyter ne sont requis que par le notebook ; les calculs restent ici.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import uuid

import numpy as np

from .labels import task_target
from .metrics import multilabel_metrics, resolve_thresholds, select_multilabel_thresholds

LABELS = ("RR", "RH")


def _json(path):
    def invalid(value):
        raise ValueError(f"Constante JSON non finie : {value}")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)
        if not isinstance(value, dict):
            raise ValueError("Objet JSON requis")
        return value
    except (OSError, ValueError) as error:
        raise ValueError(f"Impossible de lire {path}: {error}") from error


def _boolean(value):
    text = str(value).lower()
    if text not in {"true", "false", "0", "1", "0.0", "1.0"}:
        raise ValueError(f"Booléen invalide : {value!r}")
    return text in {"true", "1", "1.0"}


@dataclass
class Predictions:
    cluster_ids: np.ndarray
    keys: np.ndarray
    targets: np.ndarray
    probabilities: np.ndarray
    mask: np.ndarray
    has_chandra: np.ndarray

    def subset(self, indices):
        return Predictions(*(getattr(self, name)[indices] for name in self.__dataclass_fields__))

    def cohort(self, name="all"):
        if name not in {"all", "paired", "raw_only"}:
            raise ValueError("Cohorte attendue : all, paired ou raw_only")
        return self if name == "all" else self.subset(self.has_chandra if name == "paired" else ~self.has_chandra)

    def scores(self, thresholds=None):
        return multilabel_metrics(self.targets, self.probabilities, target_mask=self.mask, thresholds=thresholds)


@dataclass
class MultilabelRun:
    path: Path
    config: dict
    metadata: dict
    metrics: dict
    splits: dict
    history: list
    predictions: dict
    thresholds: dict
    issues: list = field(default_factory=list)

    @property
    def id(self):
        return self.path.name

    def summary(self):
        params = self.metadata.get("scattering_config") or self.config.get("model_params", {})
        split_config = self.splits.get("split_config", {})
        row = {"run_id": self.id, "path": str(self.path),
               **{k: self.config.get(k) for k in ("model", "mode", "seed", "candidate_policy")},
               "best_epoch": self.metrics.get("best_epoch"), "seconds": self.metrics.get("seconds"),
               "fold_index": split_config.get("fold_index") if split_config.get("n_folds") else None, "n_folds": split_config.get("n_folds"),
               "J": params.get("J"), "L": params.get("L"), "max_order": params.get("max_order"),
               "threshold_method": self.metrics.get("threshold_selection", {}).get("method", "fixed_legacy"),
               "threshold_RR": self.thresholds["RR"], "threshold_RH": self.thresholds["RH"]}
        # Actual resolved parameters are present in the scattering identity, not
        # necessarily in model_params. Do not invent defaults for legacy files.
        for split in ("train", "validation", "test"):
            row[f"n_{split}"] = len(self.splits[split])
        return row


def _read_predictions(path, split, manifest, config, thresholds, issues):
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or [])
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"Fichier requis absent : {path}") from error
    required = {"key", "cluster_id", "has_chandra"} | {f"{kind}_{label}" for kind in ("label", "probability") for label in LABELS}
    if not rows or not required <= fields:
        raise ValueError(f"{path}: CSV RR/RH vide ou colonnes manquantes : {sorted(required-fields)}")
    keys = np.array([r["key"] for r in rows])
    ids = np.array([r["cluster_id"] for r in rows])
    if any(not v for v in keys) or any(not v for v in ids) or len(set(keys)) != len(keys) or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: identifiants vides ou clusters dupliqués (augmentations interdites)")
    if set(keys) != set(manifest[split]):
        raise ValueError(f"{path}: clés incompatibles avec splits.json/{split}")
    samples = manifest.get("samples", {})
    target = np.array([[float(row[f"label_{name}"]) if row[f"label_{name}"] else np.nan for name in LABELS] for row in rows])
    proba = np.array([[float(row[f"probability_{name}"]) for name in LABELS] for row in rows])
    has = np.array([_boolean(row["has_chandra"]) for row in rows])
    known_fields = {f"known_{n}" for n in LABELS}
    if fields & known_fields and not known_fields <= fields:
        raise ValueError(f"{path}: masque partiel dans les colonnes du CSV")
    if known_fields <= fields:
        mask = np.array([[_boolean(row[f"known_{n}"]) for n in LABELS] for row in rows])
    else:
        derived = []
        for key in keys:
            sample = samples.get(key, {})
            if "labels" in sample and config.get("candidate_policy") in {"positive", "ignore"}:
                derived.append(task_target(sample, "rr_rh", config["candidate_policy"])[1])
            elif config.get("candidate_policy") == "positive":
                derived.append([1, 1])
            else:
                raise ValueError(f"{path}: masque absent et impossible à reconstruire sans politique/annotations")
        mask = np.array(derived, dtype=bool)
        issues.append(f"{split}: ancien CSV sans known_RR/RH ; masque dérivé de la politique enregistrée, sans certifier l'exhaustivité du catalogue.")
    scores = multilabel_metrics(target, proba, target_mask=mask, thresholds=thresholds)
    predicted = proba >= np.array(list(thresholds.values()))
    for i, row in enumerate(rows):
        item = samples.get(keys[i])
        if item is None or str(item.get("cluster_id")) != ids[i]:
            raise ValueError(f"{path}: cluster_id incompatible avec le manifeste pour {keys[i]}")
        if "chandra_path" in item and bool(item["chandra_path"]) != has[i]:
            raise ValueError(f"{path}: disponibilité CHANDRA contradictoire")
        if "labels" in item and config.get("candidate_policy") in {"positive", "ignore"}:
            expected, expected_mask = task_target(item, "rr_rh", config["candidate_policy"])
            if np.any(mask[i] & (~expected_mask.astype(bool))) or not np.array_equal(target[i][mask[i]], expected[mask[i]]):
                raise ValueError(f"{path}: vérité/masque incompatible avec les annotations de {keys[i]}")
        for j, name in enumerate(LABELS):
            if f"prediction_{name}" in fields and float(row[f"prediction_{name}"]) != int(predicted[i, j]):
                raise ValueError(f"{path}: prediction_{name} incompatible avec les seuils enregistrés")
    if any(v["n_positive"] < 5 or v["n_negative"] < 5 for v in scores["per_label"].values()):
        issues.append(f"{split}: moins de 5 positifs ou négatifs connus pour au moins un label ; estimations fragiles.")
    return Predictions(ids, keys, target, proba, mask, has)


def load_run(directory):
    """Charge schémas actuels et anciens CSV RR/RH explicitement reconnus."""
    path = Path(directory).expanduser().resolve()
    metadata, metrics, splits = (_json(path / name) for name in ("config.json", "metrics.json", "splits.json"))
    metadata_config, result_config = metadata.get("config", metadata), metrics.get("config", {})
    if not isinstance(metadata_config, dict) or not isinstance(result_config, dict):
        raise ValueError(f"{path.name}: config doit être un objet JSON")
    config = {**metadata_config, **result_config}
    if config.get("task") != "rr_rh":
        raise ValueError(f"{path.name}: seule la tâche rr_rh est acceptée, sans inférence depuis DE/NDE")
    if config.get("candidate_policy") not in {"positive", "ignore"}:
        raise ValueError(f"{path.name}: politique de candidats absente ou non prise en charge")
    status_path = path / "status.json"
    if status_path.exists() and _json(status_path).get("status") != "completed":
        raise ValueError(f"{path.name}: expérience non terminée")
    issues = []
    names = metrics.get("class_names", metadata.get("class_names"))
    if names not in (list(LABELS), list(reversed(LABELS))):
        if names == ["DE", "NDE"]:
            issues.append("Ancien class_names DE/NDE erroné : les colonnes explicites RR/RH du CSV font foi.")
        else:
            raise ValueError(f"{path.name}: noms de sorties RR/RH absents ou inconnus")
    issues.append("Convention enregistrée/historique : annotation absente = négative ; cela ne prouve pas un négatif astrophysique fiable. Aucun label n'est réinterprété ici.")
    selected = metrics.get("threshold_selection", metadata.get("evaluation", {}))
    if not isinstance(selected, dict):
        raise ValueError(f"{path.name}: provenance des seuils invalide")
    if selected.get("method") not in (None, "fixed") and selected.get("selection_split") != "validation":
        raise ValueError(f"{path.name}: seuils optimisés hors validation ou provenance absente")
    thresholds = resolve_thresholds(selected.get("thresholds", config.get("thresholds")))
    if not selected and "thresholds" not in config:
        issues.append("Seuils absents de l'ancien schéma : convention historique explicite RR=RH=0.5.")
    samples = splits.get("samples")
    if not isinstance(samples, dict):
        raise ValueError(f"{path.name}: splits.json.samples requis pour contrôler les clusters")
    groups = []
    for split in ("train", "validation", "test"):
        keys = splits.get(split)
        if not isinstance(keys, list) or not keys or len(keys) != len(set(keys)):
            raise ValueError(f"{path.name}: split {split} vide, absent ou dupliqué")
        if any(k not in samples or "cluster_id" not in samples[k] for k in keys):
            raise ValueError(f"{path.name}: manifeste incomplet pour {split}")
        ids = [str(samples[k]["cluster_id"]) for k in keys]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path.name}: clusters dupliqués dans {split}")
        groups.append(set(ids))
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError(f"{path.name}: fuite de clusters entre les splits")
    history_path = path / "history.jsonl"
    history = []
    if history_path.exists():
        for line in history_path.read_text().splitlines():
            record = json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"NaN dans history : {v}")))
            if not isinstance(record, dict) or "epoch" not in record:
                raise ValueError(f"{history_path}: champ epoch absent")
            history.append(record)
    else:
        issues.append("history.jsonl absent : les courbes d'apprentissage ne seront pas disponibles.")
    predictions = {split: _read_predictions(path / f"{split}_predictions.csv", split, splits, config, thresholds, issues)
                   for split in ("validation", "test")}
    return MultilabelRun(path, config, metadata, metrics, splits, history, predictions, thresholds, issues)


def discover_runs(root):
    """Découverte RR/RH uniquement ; ne masque pas les erreurs des runs sélectionnés."""
    root = Path(root)
    if not root.exists():
        raise ValueError(f"Dossier de résultats absent : {root}. Renseigner RUN_DIRS/RESULTS_ROOT ou utiliser le mode synthétique.")
    paths = []
    for path in sorted(root.rglob("metrics.json")):
        value = _json(path)
        if value.get("config", {}).get("task") == "rr_rh":
            paths.append(path.parent)
    if not paths:
        raise ValueError(f"Aucune expérience rr_rh trouvée dans {root}")
    return paths


def cohort_signature(run, cohort="all"):
    p = run.predictions["test"].cohort(cohort)
    if not len(p.keys):
        raise ValueError(f"{run.id}: cohorte {cohort} vide")
    entries = sorted((str(cid), [int(t[j]) if m[j] else None for j in range(2)], m.astype(int).tolist(), bool(has))
                     for cid, t, m, has in zip(p.cluster_ids, p.targets, p.mask, p.has_chandra))
    payload = {"task": "rr_rh", "candidate_policy": run.config["candidate_policy"],
               "annotation_convention": run.metrics.get("annotation_convention", "unannotated_negative"),
               "samples": entries}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def assert_comparable(runs, cohort="all"):
    if not runs:
        raise ValueError("Aucune expérience à comparer")
    signatures = [cohort_signature(run, cohort) for run in runs]
    if len(set(signatures)) != 1:
        raise ValueError("Cohortes incompatibles : clusters test, vérités, masques, couverture ou politique différents. Séparer les folds.")
    return signatures[0]


def comparison_groups(runs, cohort="all"):
    groups = {}
    for run in runs:
        groups.setdefault(cohort_signature(run, cohort), []).append(run)
    return groups


def annotation_rows(run):
    """Une ligne par cluster et split ; aucun comptage des variantes augmentées."""
    rows = []
    for split in ("train", "validation", "test"):
        predicted = run.predictions.get(split)
        lookup = {} if predicted is None else {key: i for i, key in enumerate(predicted.keys)}
        for key in run.splits[split]:
            info = run.splits["samples"][key]
            labels = set(info["labels"]) if "labels" in info else None
            if predicted is not None:
                i = lookup[key]
                target, mask = predicted.targets[i], predicted.mask[i]
                has = bool(predicted.has_chandra[i])
            elif labels is not None:
                target, mask = task_target(info, "rr_rh", run.config["candidate_policy"])
                mask = mask.astype(bool)
                has = bool(info.get("chandra_path"))
            else:
                target, mask, has = [None, None], [False, False], bool(info.get("chandra_path"))
            row = {"run_id": run.id, "split": split, "cluster_id": info["cluster_id"],
                   "has_chandra": has, "annotation_detail_available": labels is not None,
                   "combination": "".join(str(int(t)) for t in target) if all(mask) else "unknown"}
            for j, name in enumerate(LABELS):
                row[f"label_{name}"] = int(target[j]) if mask[j] else None
                row[f"known_{name}"] = bool(mask[j])
                row[f"confirmed_{name}"] = name in labels if labels is not None else None
                row[f"candidate_{name}"] = "c" + name in labels if labels is not None else None
            rows.append(row)
    return rows


def annotation_summary(run):
    rows = annotation_rows(run)
    summary = []
    for split in ("train", "validation", "test"):
        subset = [row for row in rows if row["split"] == split]
        for name in LABELS:
            truth = [r[f"label_{name}"] for r in subset if r[f"known_{name}"]]
            summary.append({"run_id": run.id, "split": split, "label": name,
                "n_clusters": len(subset), "n_known": len(truth), "n_positive": sum(truth),
                "n_negative": len(truth)-sum(truth), "prevalence": float(np.mean(truth)) if truth else None,
                "n_chandra": sum(r["has_chandra"] for r in subset),
                "annotation_detail_available": all(r["annotation_detail_available"] for r in subset)})
    return summary


def flatten_metrics(metrics):
    row = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
    for label, values in metrics["per_label"].items():
        row.update({f"{label}_{k}": v for k, v in values.items() if not isinstance(v, (dict, list))})
    return row


def validation_selection(run, *, method="f1", grid=None, min_precision=.8, thresholds=None, cohort="all"):
    p = run.predictions["validation"].cohort(cohort)
    selection = select_multilabel_thresholds(p.targets, p.probabilities, target_mask=p.mask,
        method=method, grid=grid, min_precision=min_precision, fallback=thresholds or run.thresholds)
    return {**selection, "validation_cohort": cohort, "run_id": run.id}


def threshold_curves(run, grid=None):
    p = run.predictions["validation"]
    grid = np.arange(5, 100, 5) / 100 if grid is None else grid
    rows = []
    for threshold in grid:
        scores = multilabel_metrics(p.targets, p.probabilities, target_mask=p.mask, threshold=float(threshold))
        rows.extend({"label": name, "threshold": float(threshold), **{k: scores["per_label"][name][k] for k in ("precision", "recall", "f1", "n_known")}} for name in LABELS)
    return rows


def error_rows(predictions, thresholds):
    thresholds = resolve_thresholds(thresholds)
    predicted = predictions.probabilities >= np.array(list(thresholds.values()))
    rows = []
    for i, cid in enumerate(predictions.cluster_ids):
        for j, label in enumerate(LABELS):
            if predictions.mask[i, j] and predicted[i, j] != predictions.targets[i, j]:
                row = {"cluster_id": str(cid), "key": str(predictions.keys[i]), "label": label,
                       "error": "FP" if predicted[i, j] else "FN", "has_chandra": bool(predictions.has_chandra[i])}
                for k, name in enumerate(LABELS):
                    row[f"probability_{name}"] = float(predictions.probabilities[i, k])
                    row[f"label_{name}"] = int(predictions.targets[i, k]) if predictions.mask[i, k] else None
                    row[f"known_{name}"] = bool(predictions.mask[i, k])
                rows.append(row)
    return rows


def combination_scores(predictions, thresholds, min_support=5):
    known = predictions.mask.all(1)
    rows = []
    for combo in ("00", "10", "01", "11"):
        selected = known & (predictions.targets == np.array(list(map(int, combo)))).all(1)
        n = int(selected.sum())
        rows.append({"combination": combo, "n_clusters": n, "small_support": n < min_support,
                     "subset_accuracy": predictions.subset(selected).scores(thresholds)["subset_accuracy"] if n >= min_support else None})
    rows.append({"combination": "unknown", "n_clusters": int((~known).sum()), "small_support": None, "subset_accuracy": None})
    return rows


def bootstrap_clusters(predictions, thresholds=None, *, iterations=200, seed=42, confidence=.95):
    """Percentile bootstrap de clusters uniques ; seuils figés, sans réoptimisation.

    Les réplications dégénérées sont exclues métrique par métrique ; n_valid est
    exporté. Ce bootstrap descriptif ne capture pas la sélection du modèle/seuil.
    """
    n = len(predictions.cluster_ids)
    if not n or n != len(set(predictions.cluster_ids)):
        raise ValueError("Bootstrap : clusters uniques non vides requis (pas de variantes augmentées)")
    if not isinstance(iterations, int) or iterations < 2 or not 0 < confidence < 1:
        raise ValueError("Bootstrap : iterations >= 2 et confidence entre 0 et 1 requis")
    estimates = flatten_metrics(predictions.scores(thresholds))
    wanted = [key for key in estimates if key in {"macro_f1", "micro_f1", "average_precision_macro", "roc_auc_macro", "subset_accuracy", "hamming_accuracy", "balanced_accuracy"}
              or any(key == f"{name}_{metric}" for name in LABELS for metric in ("f1", "average_precision", "roc_auc", "precision", "recall", "brier"))]
    samples = {key: [] for key in wanted}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        metrics = flatten_metrics(predictions.subset(rng.integers(0, n, size=n)).scores(thresholds))
        for key in wanted:
            if metrics[key] is not None:
                samples[key].append(metrics[key])
    alpha = (1-confidence)/2
    return [{"metric": key, "estimate": estimates[key],
             "lower": float(np.quantile(values, alpha)) if len(values) >= 2 else None,
             "upper": float(np.quantile(values, 1-alpha)) if len(values) >= 2 else None,
             "n_valid": len(values), "iterations": iterations, "seed": seed, "n_clusters": n}
            for key, values in samples.items()]


def fold_summary(runs, cohort="all", thresholds_by_run=None):
    """Moyenne et écart-type descriptifs pour vrais folds documentés et disjoints.

    Les répétitions d'un fold et les seeds/protocoles différents ne sont pas
    fusionnés. Aucun écart-type inter-fold n'est présenté comme IC indépendant.
    """
    groups = {}
    for run in runs:
        split = run.splits.get("split_config", {})
        if split.get("n_folds") is None:
            continue
        excluded = {"name", "processed_dir", "verbose"}
        if run.config.get("threshold_selection", "fixed") != "fixed":
            excluded.add("thresholds")
        protocol = {k: v for k, v in run.config.items() if k not in excluded}
        key = json.dumps({"config": protocol, "n_folds": split["n_folds"], "split_seed": split.get("split_seed")}, sort_keys=True)
        groups.setdefault(key, []).append(run)
    rows = []
    for group in groups.values():
        if len(group) < 2:
            continue
        folds = [r.splits["split_config"]["fold_index"] for r in group]
        if len(set(folds)) != len(folds):
            raise ValueError("Folds dupliqués : sélectionner une répétition par fold pour le résumé")
        sets = [set(r.predictions["test"].cluster_ids) for r in group]
        if any(a & b for i, a in enumerate(sets) for b in sets[i+1:]):
            raise ValueError("Les jeux de test des folds se chevauchent")
        thresholds_by_run = thresholds_by_run or {}
        scores = [r.predictions["test"].cohort(cohort).scores(thresholds_by_run.get(r.id, r.thresholds)) for r in group]
        for metric in ("macro_f1", "micro_f1", "average_precision_macro", "roc_auc_macro"):
            values = [m[metric] for m in scores if m[metric] is not None]
            rows.append({"model": group[0].config.get("model"), "mode": group[0].config.get("mode"),
                "metric": metric, "n_folds_present": len(group), "n_folds_expected": group[0].splits["split_config"]["n_folds"],
                "n_valid": len(values), "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    return rows


def create_output_directory(root):
    path = Path(root) / (datetime.now(timezone.utc).strftime("analysis-%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def export_rows(path, rows):
    """CSV sans remplacement implicite, valeurs indéfinies représentées vides."""
    path = Path(path)
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_score_curves(predictions, title=""):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    metrics = predictions.scores()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for j, label in enumerate(LABELS):
        selected = predictions.mask[:, j]
        y, p = predictions.targets[selected, j], predictions.probabilities[selected, j]
        scores = metrics["per_label"][label]
        pr, roc = axes[0, j], axes[1, j]
        if scores["average_precision"] is not None:
            precision, recall, _ = precision_recall_curve(y, p)
            pr.plot(recall, precision, label=f"AP = {scores['average_precision']:.3f}")
            fpr, tpr, _ = roc_curve(y, p)
            roc.plot(fpr, tpr, label=f"AUC = {scores['roc_auc']:.3f}")
        else:
            for ax in (pr, roc):
                ax.text(.5, .5, "Deux classes connues requises", ha="center", transform=ax.transAxes)
        if scores["prevalence"] is not None:
            pr.axhline(scores["prevalence"], ls="--", color="gray", label=f"Prévalence = {scores['prevalence']:.2f}")
        roc.plot([0, 1], [0, 1], "--", color="gray", label="Aléatoire")
        pr.set(xlabel="Rappel", ylabel="Précision", title=f"{label} — Precision–recall")
        roc.set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs", title=f"{label} — ROC")
        for ax in (pr, roc):
            ax.set(xlim=(0, 1), ylim=(0, 1.05))
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="best")
            ax.grid(alpha=.2)
    fig.suptitle(title)
    return fig


def plot_confusions(predictions, thresholds, title=""):
    import matplotlib.pyplot as plt
    scores = predictions.scores(thresholds)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for name, ax in zip(LABELS, axes):
        matrix = np.asarray(scores["per_label"][name]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for (i, j), value in np.ndenumerate(matrix):
            tag = [["TN", "FP"], ["FN", "TP"]][i][j]
            ax.text(j, i, f"{tag}\n{value}", ha="center", va="center", color="white" if value > matrix.max()/2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Prédiction", ylabel="Vérité connue",
               title=f"{name} — seuil {scores['per_label'][name]['threshold']:.2f}")
    fig.suptitle(title)
    return fig


def plot_probabilities(predictions, thresholds, title=""):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for j, (name, ax) in enumerate(zip(LABELS, axes)):
        for value, text in ((0, "Négatifs connus"), (1, "Positifs connus")):
            selected = predictions.mask[:, j] & (predictions.targets[:, j] == value)
            ax.hist(predictions.probabilities[selected, j], bins=np.linspace(0, 1, 11), alpha=.55, label=f"{text} (n={selected.sum()})")
        ax.axvline(thresholds[name], ls="--", color="black", label="Seuil figé")
        ax.set(title=name, xlabel="Probabilité sigmoid", ylabel="Nombre de clusters")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    return fig


def plot_learning_history(run):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for metric, ax in zip(("loss", "macro_f1", "average_precision_macro", "roc_auc_macro"), axes.flat):
        found = False
        for split in ("train", "validation"):
            pairs = [(record["epoch"], record.get(split, {}).get(metric)) for record in run.history]
            pairs = [(x, y) for x, y in pairs if y is not None]
            if pairs:
                ax.plot(*zip(*pairs), marker="o", label=split)
                found = True
        if not found:
            ax.text(.5, .5, "Champ absent / non défini", ha="center", transform=ax.transAxes)
        else:
            ax.legend()
        ax.set(title=metric, xlabel="Époque")
        ax.grid(alpha=.2)
    fig.suptitle(f"{run.id} — seuils d'entraînement, avant sélection finale")
    return fig


def plot_validation_thresholds(run, selected):
    import matplotlib.pyplot as plt
    rows = threshold_curves(run)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, ax in zip(LABELS, axes):
        subset = [row for row in rows if row["label"] == name]
        for metric in ("precision", "recall", "f1"):
            ax.plot([r["threshold"] for r in subset], [r[metric] for r in subset], label=metric)
        ax.axvline(selected[name], color="black", ls="--", label="Seuil appliqué")
        ax.set(title=f"{name} — validation uniquement", xlabel="Seuil", ylabel="Score", ylim=(0, 1.05))
        ax.legend()
        ax.grid(alpha=.2)
    return fig


def plot_comparison(rows, intervals):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    names = [r["run_id"] for r in rows]
    x = np.arange(len(names))
    for metric, ax in zip(("macro_f1", "average_precision_macro", "roc_auc_macro"), axes):
        values = [r.get(metric) for r in rows]
        ax.scatter(x, [np.nan if v is None else v for v in values], label="Estimation test")
        for i, name in enumerate(names):
            ci = next((v for v in intervals.get(name, []) if v["metric"] == metric), None)
            if ci and ci["lower"] is not None:
                ax.vlines(i, ci["lower"], ci["upper"], color="C0")
                ax.hlines([ci["lower"], ci["upper"]], i-.07, i+.07, color="C0")
        ax.set(title=metric, xticks=x, xticklabels=[f"{r['model']}\n{r['mode']}\n{r['run_id'][-8:]}" for r in rows], ylim=(-.03, 1.03))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Même cohorte test — IC bootstrap de clusters à seuils figés")
    return fig


def create_demo_runs(root, seed=42):
    """Artefacts SYNTHÉTIQUES au schéma du dépôt ; aucun entraînement prétendu.

    Annotations explicites, candidats masqués et probabilités simulées. Répertoire
    nouveau obligatoire ; aucun fichier utilisateur n'est requis ni écrasé.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    patterns = [{"RR"}, {"RH"}, {"RR", "RH"}, {"cRR"}, {"cRH"}, {"cRR", "cRH"}]
    keys = [f"synthetic-{i}.fits" for i in range(48)]
    samples = {key: {"cluster_id": key.removesuffix(".fits"), "labels": sorted(patterns[i % 6]),
                     "raw_path": f"synthetic/{key}", "chandra_path": f"synthetic/chandra-{key}" if i % 3 else None}
               for i, key in enumerate(keys)}
    splits = {"train": keys[:24], "validation": keys[24:36], "test": keys[36:], "samples": samples}
    paths = []
    for model in ("image_cnn", "dual_ssn", "dual_encoder_ssn"):
        path = root / f"synthetic-{model}"
        path.mkdir()
        config = {"task": "rr_rh", "name": path.name, "model": model, "mode": "multimodal", "seed": seed,
                  "candidate_policy": "ignore", "thresholds": {"RR": .5, "RH": .5},
                  "model_params": {"J": 2, "L": 2, "max_order": 2} if "ssn" in model else {}}
        data, scores = {}, {}
        for split in ("validation", "test"):
            rows = []
            y, masks = [], []
            for key in splits[split]:
                target, mask = task_target(samples[key], "rr_rh", "ignore")
                y.append(target)
                masks.append(mask)
            probabilities = np.clip(.15 + .5*np.asarray(y) + rng.normal(0, .23, (len(y), 2)), .01, .99)
            for key, target, mask, proba in zip(splits[split], y, masks, probabilities):
                row = {"key": key, "cluster_id": samples[key]["cluster_id"], "has_chandra": bool(samples[key]["chandra_path"])}
                for j, label in enumerate(LABELS):
                    row.update({f"label_{label}": int(target[j]), f"known_{label}": int(mask[j]),
                                f"probability_{label}": float(proba[j]), f"prediction_{label}": int(proba[j] >= .5)})
                rows.append(row)
            export_rows(path / f"{split}_predictions.csv", rows)
            scores[split] = multilabel_metrics(y, probabilities, target_mask=masks)
        metrics = {"schema_version": 2, "run_id": path.name, "class_names": list(LABELS), "config": config,
                   "synthetic": True, "annotation_convention": "unannotated_negative", "best_epoch": None,
                   "seconds": None, "validation": scores["validation"], "test": scores["test"]}
        for name, value in (("config.json", {"config": config, "synthetic": True}), ("metrics.json", metrics),
                            ("splits.json", splits), ("status.json", {"status": "completed"})):
            (path/name).write_text(json.dumps(value, indent=2, allow_nan=False))
        # Empty history deliberately exercises missing training curves. These
        # scores are simulated predictions, never fabricated training epochs.
        (path/"history.jsonl").write_text("")
        paths.append(path)
    return paths
