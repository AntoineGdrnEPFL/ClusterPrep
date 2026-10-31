"""Réutilisation des splits historiques pour le transfer learning."""

import json
from collections import Counter
from pathlib import Path
import numpy as np

from .splits import eligible_for_task

SPLIT_NAMES = ("train", "validation", "test")


def count_label_combinations(keys, data_info):
    return dict(Counter(tuple(sorted(data_info[key]["labels"])) for key in keys))


def count_target_combinations(keys, data_info):
    counts = Counter()

    for key in keys:
        labels = set(data_info[key]["labels"])

        if "NDE" in labels:
            target = "00"
        else:
            rr = bool(labels & {"RR", "cRR"})
            rh = bool(labels & {"RH", "cRH"})
            target = f"{int(rr)}{int(rh)}"

        counts[target] += 1

    return dict(counts)


def distribute_new_clusters(
    new_keys,
    historical_splits,
    seed=42,
):
    rng = np.random.default_rng(seed)

    new_keys = list(new_keys)
    rng.shuffle(new_keys)

    historical_sizes = np.array(
        [len(s) for s in historical_splits],
        dtype=float,
    )

    proportions = historical_sizes / historical_sizes.sum()

    expected = proportions * len(new_keys)
    allocation = np.floor(expected).astype(int)

    remaining = len(new_keys) - allocation.sum()

    fractional = expected - allocation

    for index in np.argsort(-fractional)[:remaining]:
        allocation[index] += 1

    result = []
    offset = 0

    for split, count in zip(historical_splits, allocation):
        additions = new_keys[offset : offset + count]
        result.append(list(split) + additions)
        offset += count

    return tuple(result)


def _read_source_splits(source_experiment):
    """Lit splits.json à partir d'un répertoire d'expérience."""
    source_dir = Path(source_experiment)
    path = source_dir / "splits.json"

    if not path.is_file():
        raise FileNotFoundError(path)

    manifest = json.loads(path.read_text(encoding="utf-8"))

    for name in SPLIT_NAMES:
        if name not in manifest:
            raise ValueError(f"Split historique absent : {name}")
        if not isinstance(manifest[name], list):
            raise ValueError(f"Split {name} invalide")

    if not isinstance(manifest.get("samples"), dict):
        raise ValueError(
            "Le manifest doit contenir un dictionnaire samples "
            "associant chaque clé à son cluster_id."
        )

    return manifest


def audit_transfer_splits(source_experiment, data_info, splits):
    """Audit target holdouts against supervised source exposure by cluster ID."""
    manifest = _read_source_splits(source_experiment)
    source = {}
    seen = set()
    for name in SPLIT_NAMES:
        ids = []
        for key in manifest[name]:
            cid = manifest["samples"].get(key, {}).get("cluster_id")
            if cid is None:
                raise ValueError(f"Source cluster_id missing: {key}")
            cid = str(cid)
            if cid in seen:
                raise ValueError(f"Duplicate source cluster_id: {cid}")
            seen.add(cid)
            ids.append(cid)
        source[name] = set(ids)
    target = {name: {str(data_info[k]["cluster_id"]) for k in keys}
              for name, keys in zip(SPLIT_NAMES, splits)}
    for name in ("validation", "test"):
        overlap = sorted(target[name] & source["train"])
        if overlap:
            raise ValueError(f"Fuite : target {name} overlaps supervised source train: {overlap}")
    validation_overlap = sorted(target["test"] & source["validation"])
    return {
        "target_test_in_source_train": [],
        "target_validation_in_source_train": [],
        "target_test_in_source_validation": validation_overlap,
        "source_validation_used_for_checkpoint_selection": True,
        "warnings": (["Target test includes source validation clusters used for checkpoint selection"]
                     if validation_overlap else []),
        "source_cluster_ids": {name: sorted(ids) for name, ids in source.items()},
    }


def _build_cluster_index(data_info):
    """
    Indexe le catalogue actuel par identifiant de cluster.

    Refuse les identifiants ambigus : un même cluster_id
    ne doit correspondre qu'à une entrée canonique.
    """
    index = {}

    for key, info in data_info.items():
        cluster_id = str(info["cluster_id"])

        if cluster_id in index:
            raise ValueError(
                f"Cluster dupliqué dans le catalogue actuel : "
                f"{cluster_id} ({index[cluster_id]}, {key})"
            )

        index[cluster_id] = key

    return index


def _validate_disjoint_splits(splits, data_info):
    """Vérifie l'absence de clusters communs entre les splits."""
    seen = {}

    for name, keys in zip(SPLIT_NAMES, splits):
        if len(keys) != len(set(keys)):
            raise ValueError(f"Clés dupliquées dans {name}")

        for key in keys:
            cluster_id = data_info[key]["cluster_id"]

            if cluster_id in seen:
                raise ValueError(
                    f"Fuite de données : {cluster_id} appartient "
                    f"à {seen[cluster_id]} et {name}"
                )

            seen[cluster_id] = name


def resolve_transfer_splits(
    source_experiment,
    data_info,
    task="rr_rh",
    mode="raw",
    candidate_policy="positive",
    allow_subset=False,
    add_new_clusters=True,
    new_cluster_labels=("cRR", "cRH"),
    seed=42,
):
    """
    Reconstruit les splits historiques et peut les étendre
    avec de nouveaux amas, sans déplacer les anciens.

    Returns
    -------
    tuple
        ((train_keys, validation_keys, test_keys), report)
    """
    if mode not in {"raw", "pair", "multimodal"}:
        raise ValueError(f"Mode inconnu : {mode}")

    if task not in {"de_nde", "rr_rh", "rr_binary", "rh_binary"}:
        raise ValueError(f"Tâche inconnue : {task}")

    if candidate_policy not in {"positive", "ignore"} or (task not in {"rr_binary", "rh_binary"} and candidate_policy != "positive"):
        raise ValueError(
            "Cette version prend uniquement en charge " "candidate_policy='positive'."
        )

    manifest = _read_source_splits(source_experiment)
    samples = manifest["samples"]
    cluster_index = _build_cluster_index(data_info)

    report = {
        "source_experiment": str(Path(source_experiment).resolve()),
        "task": task,
        "mode": mode,
        "candidate_policy": candidate_policy,
        "split_policy": (
            "preserve_historical_and_extend"
            if add_new_clusters
            else "preserve_historical_only"
        ),
        "seed": seed,
        "new_cluster_labels": list(new_cluster_labels),
        "allocation_method": ("distribute_new_clusters" if add_new_clusters else None),
        "splits": {},
        "excluded": [],
        "added_clusters": {},
        "source_file_changes": [],
        "legacy_without_hash": [],
    }

    def validate_annotations(info):
        cluster_id = str(info["cluster_id"])
        labels = set(info["labels"])

        if "NDE" in labels and labels.intersection({"DE", "RR", "RH", "cRR", "cRH"}):
            raise ValueError(f"Annotations contradictoires : {cluster_id}")

    def is_eligible(info):
        validate_annotations(info)

        if not eligible_for_task(info, task, candidate_policy):
            return False

        if mode == "pair" and not info.get("chandra_path"):
            return False

        return True

    def distribution(keys):
        """Statistiques sur les amas uniques, sans augmentation."""
        result = {
            "labels": dict(
                Counter(label for key in keys for label in data_info[key]["labels"])
            ),
            "label_combinations": {
                "+".join(labels): count
                for labels, count in count_label_combinations(keys, data_info).items()
            },
        }

        if task == "rr_rh":
            # Cette fonction doit suivre candidate_policy="positive".
            counts = count_target_combinations(keys, data_info)
            result["target_combinations"] = {
                target: counts.get(target, 0) for target in ("00", "10", "01", "11")
            }

            if sum(result["target_combinations"].values()) != len(keys):
                raise ValueError(
                    "Distribution RR/RH incohérente : "
                    "tous les amas doivent avoir une cible connue."
                )

        return result

    # --------------------------------------------------
    # 1. Résolution des splits historiques
    # --------------------------------------------------

    resolved = {}
    source_seen = {}
    historical_metadata = {}

    for split_name in SPLIT_NAMES:
        source_keys = manifest[split_name]
        current_keys = []

        if len(source_keys) != len(set(source_keys)):
            raise ValueError(f"Clés répétées dans le split source {split_name}")

        for old_key in source_keys:
            if old_key not in samples:
                raise ValueError(f"Échantillon source absent de samples : {old_key}")

            old_info = samples[old_key]

            if not isinstance(old_info, dict):
                raise ValueError(f"Métadonnées invalides : {old_key}")

            cluster_id = old_info.get("cluster_id")

            if not cluster_id:
                raise ValueError(f"cluster_id manquant pour {old_key}")

            cluster_id = str(cluster_id)

            if cluster_id in source_seen:
                raise ValueError(
                    f"Cluster historique dupliqué : {cluster_id} "
                    f"dans {source_seen[cluster_id]} et {split_name}"
                )

            source_seen[cluster_id] = split_name

            if cluster_id not in cluster_index:
                report["excluded"].append(
                    {
                        "cluster_id": cluster_id,
                        "source_key": old_key,
                        "split": split_name,
                        "reason": "missing_from_current_catalog",
                    }
                )
                continue

            new_key = cluster_index[cluster_id]
            current_info = data_info[new_key]

            if not is_eligible(current_info):
                report["excluded"].append(
                    {
                        "cluster_id": cluster_id,
                        "source_key": old_key,
                        "split": split_name,
                        "reason": "ineligible_for_target_task_or_mode",
                        "labels": sorted(current_info["labels"]),
                    }
                )
                continue

            if not old_info.get("hash"):
                report["legacy_without_hash"].append(cluster_id)

            old_raw_path = old_info.get("raw_path")
            new_raw_path = current_info.get("raw_path")

            if old_raw_path and new_raw_path:
                if Path(old_raw_path).resolve() != Path(new_raw_path).resolve():
                    report["source_file_changes"].append(
                        {
                            "cluster_id": cluster_id,
                            "old_raw_path": old_raw_path,
                            "new_raw_path": new_raw_path,
                            "identity_verified": False,
                        }
                    )

            current_keys.append(new_key)

        resolved[split_name] = current_keys

        historical_metadata[split_name] = {
            "source_count": len(source_keys),
            "resolved_count": len(current_keys),
            "excluded_count": len(source_keys) - len(current_keys),
            "distribution": distribution(current_keys),
        }

    if report["excluded"] and not allow_subset:
        details = [
            f"{item['split']}: {item['cluster_id']} " f"({item['reason']})"
            for item in report["excluded"][:10]
        ]

        raise ValueError(
            "Impossible de réutiliser intégralement les "
            "splits historiques.\n"
            + "\n".join(details)
            + "\nUtiliser allow_subset=True uniquement "
            "si les exclusions sont volontaires."
        )

    historical_splits = tuple(list(resolved[name]) for name in SPLIT_NAMES)

    _validate_disjoint_splits(historical_splits, data_info)

    # IMPORTANT : utiliser TOUS les IDs historiques,
    # y compris les amas exclus, pour éviter de les
    # réintroduire en tant que nouveaux candidats.
    historical_ids = set(source_seen)

    # --------------------------------------------------
    # 2. Identification des nouveaux amas
    # --------------------------------------------------

    new_keys = []

    for key, info in data_info.items():
        cluster_id = str(info["cluster_id"])
        labels = set(info["labels"])

        if cluster_id in historical_ids:
            continue

        if not labels.intersection(new_cluster_labels):
            continue

        if not is_eligible(info):
            continue

        new_keys.append(key)

    new_keys = sorted(new_keys)

    report["new_clusters_eligible"] = len(new_keys)
    report["new_clusters_eligible_ids"] = [
        str(data_info[key]["cluster_id"]) for key in new_keys
    ]

    # --------------------------------------------------
    # 3. Extension reproductible des splits
    # --------------------------------------------------

    if add_new_clusters and new_keys:
        final_splits = distribute_new_clusters(
            new_keys,
            historical_splits,
            seed=seed,
        )
    else:
        final_splits = historical_splits

    final_splits = tuple(list(split) for split in final_splits)

    # Vérification des invariants
    for split_name, final_keys in zip(SPLIT_NAMES, final_splits):
        original_keys = set(resolved[split_name])

        if not original_keys.issubset(set(final_keys)):
            raise ValueError(
                f"Des amas historiques ont été déplacés "
                f"ou supprimés du split {split_name}."
            )

    for name, keys in zip(SPLIT_NAMES, final_splits):
        if not keys:
            raise ValueError(f"Split {name} vide après résolution.")

    _validate_disjoint_splits(final_splits, data_info)

    # --------------------------------------------------
    # 4. Rapport de provenance après extension
    # --------------------------------------------------

    all_added_keys = []

    for split_name, final_keys in zip(SPLIT_NAMES, final_splits):
        historical_keys = resolved[split_name]
        historical_key_set = set(historical_keys)

        added_keys = [key for key in final_keys if key not in historical_key_set]

        all_added_keys.extend(added_keys)

        added_details = [
            {
                "key": key,
                "cluster_id": str(data_info[key]["cluster_id"]),
                "labels": sorted(data_info[key]["labels"]),
                "origin": "new_candidate",
                "assigned_split": split_name,
            }
            for key in added_keys
        ]

        report["added_clusters"][split_name] = added_details

        report["splits"][split_name] = {
            **historical_metadata[split_name],
            "historical_count": len(historical_keys),
            "added_count": len(added_keys),
            "final_count": len(final_keys),
            "added_distribution": distribution(added_keys),
            "final_distribution": distribution(final_keys),
            "historical_cluster_ids": [
                str(data_info[key]["cluster_id"]) for key in historical_keys
            ],
            "added_cluster_ids": [
                str(data_info[key]["cluster_id"]) for key in added_keys
            ],
        }

        if len(historical_keys) + len(added_keys) != len(final_keys):
            raise ValueError(f"Effectifs incohérents pour {split_name}")

    expected_added = set(new_keys) if add_new_clusters else set()

    if len(all_added_keys) != len(set(all_added_keys)):
        raise ValueError("Un nouvel amas apparaît dans plusieurs splits.")

    if set(all_added_keys) != expected_added:
        raise ValueError(
            "L'allocation des nouveaux amas est incomplète "
            "ou contient des amas inattendus."
        )

    # --------------------------------------------------
    # 5. Synthèse globale
    # --------------------------------------------------

    report["summary"] = {
        "source_count": sum(
            info["source_count"] for info in historical_metadata.values()
        ),
        "historical_resolved_count": sum(len(resolved[name]) for name in SPLIT_NAMES),
        "historical_excluded_count": len(report["excluded"]),
        "new_eligible_count": len(new_keys),
        "new_added_count": len(all_added_keys),
        "final_count": sum(map(len, final_splits)),
    }

    # Compatibilité avec les champs existants du rapport.
    report["total_resolved"] = report["summary"]["final_count"]
    report["total_excluded"] = len(report["excluded"])

    return final_splits, report
