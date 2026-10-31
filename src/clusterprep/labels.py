"""Conventions d'annotation communes aux datasets, splits et pertes."""
import numpy as np

BINARY_TASKS = frozenset({"de_nde", "rr_binary", "rh_binary"})
SUBTYPE_TASKS = frozenset({"rr_binary", "rh_binary"})
TASK_NAMES = BINARY_TASKS | {"rr_rh"}

ANNOTATION_NAMES = ("DE", "NDE", "RR", "RH", "cRR", "cRH")


def labels_for(info):
    labels = set(info["labels"]) if "labels" in info else {info["class"]}
    if "class" in info and info["class"] not in labels:
        raise ValueError("Classes contradictoires avec labels")
    if "NDE" in labels and labels & {"DE", "RR", "RH", "cRR", "cRH"}:
        raise ValueError("Cluster ambigu DE/NDE : annotations contradictoires")
    return labels


def task_target(info, task, candidate_policy="positive", class_names=("DE", "NDE")):
    if candidate_policy not in {"positive", "ignore"}:
        raise ValueError("candidate_policy doit être positive ou ignore")
    labels = labels_for(info)
    if task == "de_nde":
        chosen = labels & {"DE", "NDE"}
        if not chosen:
            raise ValueError("Cluster non éligible DE/NDE")
        return list(class_names).index(next(iter(chosen))), None
    if task in SUBTYPE_TASKS:
        target = binary_subtype_target(info, task, candidate_policy)
        if target is None:
            raise ValueError(f"Cluster non éligible {task}: {sorted(labels)}")
        return target, None
    if task != "rr_rh":
        raise ValueError(f"Tâche inconnue : {task}")
    if not labels & {"NDE", "RR", "RH", "cRR", "cRH"}:
        raise ValueError("Cluster non éligible RR/RH")
    # NDE est un négatif confirmé ; les candidats ignorés restent masqués.
    target = np.array([name in labels or (candidate_policy == "positive" and "c" + name in labels)
                       for name in ("RR", "RH")], dtype=np.float32)
    mask = np.array([not (candidate_policy == "ignore" and "c" + name in labels and name not in labels)
                     for name in ("RR", "RH")], dtype=np.float32)
    return target, mask


def merge_cluster_labels(data_info):
    """Return a copy with the union of annotations on every alias of a cluster."""
    grouped = {}
    for info in data_info.values():
        grouped.setdefault(info["cluster_id"], set()).update(labels_for(info))
    for cid, labels in grouped.items():
        try:
            labels_for({"labels": labels})
        except ValueError as error:
            raise ValueError(f"Annotations contradictoires pour {cid}: {sorted(labels)}") from error
    return {key: {**info, "labels": grouped[info["cluster_id"]].copy()}
            for key, info in data_info.items()}


def binary_subtype_target(info, task, candidate_policy="positive"):
    """Scalar subtype target, or None when no certain task target is available."""
    if task not in SUBTYPE_TASKS or candidate_policy not in {"positive", "ignore"}:
        raise ValueError("task ou candidate_policy invalide")
    labels = labels_for(info)
    positive = "RR" if task == "rr_binary" else "RH"
    if candidate_policy == "ignore" and "c" + positive in labels and positive not in labels:
        return None
    allowed = {"RR", "RH", "NDE"}
    if candidate_policy == "positive":
        allowed |= {"cRR", "cRH"}
    if not labels & allowed:
        return None
    return int(positive in labels or (candidate_policy == "positive" and "c" + positive in labels))
