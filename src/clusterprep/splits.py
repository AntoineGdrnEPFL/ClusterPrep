"""Séparations stratifiées sans fuite entre amas."""

from collections import defaultdict, Counter
import numpy as np
import warnings
from .labels import labels_for, task_target, merge_cluster_labels, binary_subtype_target, SUBTYPE_TASKS, BINARY_TASKS
from sklearn.model_selection import StratifiedKFold, train_test_split


def eligible_for_task(info, task, candidate_policy="positive"):
    if candidate_policy not in {"positive", "ignore"}:
        raise ValueError("candidate_policy doit être 'positive' ou 'ignore'")
    labels = labels_for(info)
    if task == "de_nde":
        return bool(labels & {"DE", "NDE"})
    if task == "rr_rh":
        allowed = {"NDE", "RR", "RH"}
        if candidate_policy == "positive":
            allowed |= {"cRR", "cRH"}
        return bool(labels & allowed)
    if task in SUBTYPE_TASKS:
        return binary_subtype_target(info, task, candidate_policy) is not None

    raise ValueError(f"Tâche inconnue : {task}")


def multilabel_stratum(info, candidate_policy="positive"):
    target, mask = task_target(info, "rr_rh", candidate_policy)
    return "".join(str(int(t)) if m else "?" for t, m in zip(target, mask))


def _iterative(strata, *, seed, n_folds=None, test_size=None):
    counts = dict(Counter(strata))
    try:
        from iterstrat.ml_stratifiers import (
            MultilabelStratifiedKFold,
            MultilabelStratifiedShuffleSplit,
        )
    except ImportError as error:
        raise ValueError(
            f"Stratification RR/RH impossible avec effectifs {counts}; installer iterative-stratification"
        ) from error
    # Distinguish known positive, known negative and unknown for each label.
    y = np.asarray(
        [
            [int(s[j] == value) for j in range(2) for value in ("1", "0", "?")]
            for s in strata
        ]
    )
    if len(strata) < (n_folds or 3):
        raise ValueError(
            f"Pas assez de clusters pour la stratification RR/RH: {counts}"
        )
    warnings.warn(
        f"Stratification RR/RH itérative : combinaisons rares, effectifs {counts}"
    )
    splitter = (
        MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        if n_folds
        else MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
    )
    return list(splitter.split(np.zeros(len(y)), y))


def split_data_keys(
    data_info,
    mode="raw",
    test_size=0.20,
    val_size=0.20,
    seed=42,
    n_folds=None,
    fold_index=0,
    task="de_nde",
    candidate_policy="positive",
    use_val=True,
):
    """Sépare les données par cluster, avec un fold de test optionnel."""
    if mode not in {"raw", "pair", "multimodal"}:
        raise ValueError("mode doit être raw, pair ou multimodal")
    if not isinstance(use_val, bool):
        raise ValueError("use_val doit être booléen")
    if not 0 <= val_size < 1:
        raise ValueError("val_size doit être dans [0, 1[")
    use_val = use_val and val_size > 0
    if not 0 < test_size < 1 or (
        use_val and n_folds is None and test_size + val_size >= 1
    ):
        raise ValueError("test_size + val_size doit être < 1")
    if n_folds is not None:
        if not isinstance(n_folds, int) or n_folds < 2:
            raise ValueError("n_folds doit être un entier supérieur ou égal à 2")
        if not isinstance(fold_index, int) or not 0 <= fold_index < n_folds:
            raise ValueError(f"fold_index doit être compris entre 0 et {n_folds - 1}")

    data_info = merge_cluster_labels(data_info)
    # Sélection des observations disponibles pour l'expérience.
    eligible_keys = [
        key
        for key, info in data_info.items()
        if eligible_for_task(info, task, candidate_policy)
        and (mode != "pair" or info.get("chandra_path"))
    ]

    # Regroupement des clés par cluster.
    keys_by_cluster = defaultdict(list)
    class_by_cluster = {}

    for key in eligible_keys:
        info = data_info[key]
        cluster_id = info["cluster_id"]
        target, mask = task_target(info, task, candidate_policy)
        signature = (
            str((target.tolist(), mask.tolist())) if task == "rr_rh" else str(target)
        )

        if cluster_id in class_by_cluster and class_by_cluster[cluster_id] != signature:
            raise ValueError(f"Classes contradictoires pour {cluster_id}")

        keys_by_cluster[cluster_id].append(key)
        class_by_cluster[cluster_id] = signature

    cluster_ids = sorted(keys_by_cluster)

    if task == "de_nde":
        strata = [
            "DE" if "DE" in data_info[keys_by_cluster[cid][0]]["labels"] else "NDE"
            for cid in cluster_ids
        ]

    elif task in SUBTYPE_TASKS:
        strata = [multilabel_stratum(data_info[keys_by_cluster[cid][0]], candidate_policy) for cid in cluster_ids]
    elif task == "rr_rh":
        strata = [
            multilabel_stratum(
                data_info[keys_by_cluster[cid][0]],
                candidate_policy=candidate_policy,
            )
            for cid in cluster_ids
        ]

    else:
        raise ValueError(f"Tâche inconnue : {task}")
    if not cluster_ids:
        raise ValueError("Aucun cluster éligible")

    if task in SUBTYPE_TASKS:
        binary_strata = [class_by_cluster[cid] for cid in cluster_ids]
        required = n_folds or 2
        if min(Counter(strata).values()) < required:
            warnings.warn(f"{task}: catégories rares, stratification sur la cible binaire")
            strata = binary_strata

    if n_folds is None:
        # Séparation classique train / validation / test.
        train_val_ids, test_ids = _safe_split(
            cluster_ids,
            test_size=test_size,
            random_state=seed,
            stratify=strata,
            task=task,
        )
    else:
        # Chaque fold est constitué de clusters complets et stratifiés.
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        if min(strata.count(label) for label in set(strata)) < n_folds:
            if task in BINARY_TASKS:
                raise ValueError(
                    "Chaque classe doit contenir au moins n_folds clusters"
                )
            folds = _iterative(strata, seed=seed, n_folds=n_folds)
        else:
            folds = list(splitter.split(cluster_ids, strata))
        train_val_indices, test_indices = folds[fold_index]
        train_val_ids = [cluster_ids[i] for i in train_val_indices]
        test_ids = [cluster_ids[i] for i in test_indices]

    if not use_val:
        return (
            [key for cid in train_val_ids for key in keys_by_cluster[cid]],
            [],
            [key for cid in test_ids for key in keys_by_cluster[cid]],
        )

    # La validation est extraite du pool train, sans toucher au fold de test.
    if not train_val_ids or not cluster_ids:
        raise ValueError("Aucun cluster disponible pour l’entraînement")
    # val_size est une fraction du jeu complet, même avec un fold de test.
    relative_val_size = val_size * len(cluster_ids) / len(train_val_ids)
    if not 0 < relative_val_size < 1:
        raise ValueError("val_size trop grand pour le pool hors test")
    stratum_by_cluster = dict(zip(cluster_ids, strata))
    train_ids, val_ids = _safe_split(
        train_val_ids,
        test_size=relative_val_size,
        random_state=seed,
        stratify=[stratum_by_cluster[cid] for cid in train_val_ids],
        task=task,
    )

    def keys_for(cluster_subset):
        return [
            key for cluster_id in cluster_subset for key in keys_by_cluster[cluster_id]
        ]

    return keys_for(train_ids), keys_for(val_ids), keys_for(test_ids)


def _safe_split(ids, *, stratify, task="de_nde", **kwargs):
    try:
        return train_test_split(ids, stratify=stratify, **kwargs)
    except ValueError as error:
        if task == "rr_rh":
            try:
                a, b = _iterative(
                    stratify, seed=kwargs["random_state"], test_size=kwargs["test_size"]
                )[0]
                if not len(a) or not len(b):
                    raise ValueError("split vide")
                return [ids[i] for i in a], [ids[i] for i in b]
            except ValueError as fallback_error:
                raise ValueError(
                    f"Stratification RR/RH impossible; effectifs {dict(Counter(stratify))}: {fallback_error}"
                ) from error
        if task in SUBTYPE_TASKS:
            # Collapse original categories onto the requested scalar target.
            j = 0 if task == "rr_binary" else 1
            binary = [s[j] if len(s) == 2 else s for s in stratify]
            try:
                return train_test_split(ids, stratify=binary, **kwargs)
            except ValueError as fallback_error:
                raise ValueError(f"{task}: stratification impossible, effectifs {dict(Counter(binary))}") from fallback_error
        if len(ids) < 3:
            raise ValueError(
                "Pas assez de clusters pour trois splits non vides"
            ) from error
        warnings.warn(
            "Stratification impossible (combinaisons rares) : séparation aléatoire reproductible"
        )
        return train_test_split(ids, **kwargs)


def split_distribution(data_info, splits, task="de_nde", candidate_policy="positive"):
    info = merge_cluster_labels(data_info)
    result = {"task": task, "candidate_policy": candidate_policy, "labels": {}}
    for name, keys in zip(("train", "validation", "test"), splits):
        unique = {info[k]["cluster_id"]: info[k] for k in keys}
        counts = dict.fromkeys(
            ("00", "10", "01", "11") if task != "de_nde" else ("DE", "NDE"), 0
        )
        for item in unique.values():
            stratum = (
                multilabel_stratum(item, candidate_policy)
                if task != "de_nde"
                else ("DE" if "DE" in labels_for(item) else "NDE")
            )
            counts[stratum] = counts.get(stratum, 0) + 1
        result[name] = counts
        if task in SUBTYPE_TASKS:
            targets = [task_target(item, task, candidate_policy)[0] for item in unique.values()]
            pos = sum(targets)
            neg = len(targets) - pos
            result["labels"][name] = {"positive": pos, "negative": neg,
                                      "roc_auc_defined": bool(pos and neg)}
        if task == "rr_rh":
            pairs = [
                task_target(item, task, candidate_policy) for item in unique.values()
            ]
            result["labels"][name] = {}
            for j, label in enumerate(("RR", "RH")):
                pos = sum(int(t[j] == 1 and m[j]) for t, m in pairs)
                neg = sum(int(t[j] == 0 and m[j]) for t, m in pairs)
                result["labels"][name][label] = {
                    "positive": pos,
                    "negative": neg,
                    "unknown": len(pairs) - pos - neg,
                    "roc_auc_defined": bool(pos and neg),
                }
                if keys and (not pos or not neg):
                    warnings.warn(
                        f"{name}/{label}: ROC-AUC indéfinie ({pos} positifs, {neg} négatifs connus)"
                    )
    return result
