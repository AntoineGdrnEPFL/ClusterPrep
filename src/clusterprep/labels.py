"""Conventions d'annotation communes aux datasets, splits et pertes."""
import numpy as np

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
