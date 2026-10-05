"""Séparations stratifiées sans fuite entre amas."""
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data_keys(
    data_info,
    mode="raw",
    test_size=0.20,
    val_size=0.20,
    seed=42,
    n_folds=None,
    fold_index=0,
):
    """Sépare les données par cluster, avec un fold de test optionnel."""
    if mode not in {"raw", "pair", "multimodal"}:
        raise ValueError("mode doit être raw, pair ou multimodal")
    if not 0 < test_size < 1 or not 0 < val_size < 1 or (n_folds is None and test_size + val_size >= 1):
        raise ValueError("test_size + val_size doit être < 1")
    if n_folds is not None:
        if not isinstance(n_folds, int) or n_folds < 2:
            raise ValueError("n_folds doit être un entier supérieur ou égal à 2")
        if not isinstance(fold_index, int) or not 0 <= fold_index < n_folds:
            raise ValueError(
                f"fold_index doit être compris entre 0 et {n_folds - 1}"
            )

    # Sélection des observations disponibles pour l'expérience.
    eligible_keys = [
        key for key, info in data_info.items()
        if mode != "pair" or info.get("chandra_path")
    ]

    # Regroupement des clés par cluster.
    keys_by_cluster = defaultdict(list)
    class_by_cluster = {}

    for key in eligible_keys:
        info = data_info[key]
        cluster_id = info["cluster_id"]

        if (
            cluster_id in class_by_cluster
            and class_by_cluster[cluster_id] != info["class"]
        ):
            raise ValueError(
                f"Classes contradictoires pour {cluster_id}"
            )

        keys_by_cluster[cluster_id].append(key)
        class_by_cluster[cluster_id] = info["class"]

    cluster_ids = sorted(keys_by_cluster)
    labels = [class_by_cluster[cid] for cid in cluster_ids]
    if not cluster_ids:
        raise ValueError("Aucun cluster éligible")

    if n_folds is None:
        # Séparation classique train / validation / test.
        train_val_ids, test_ids = train_test_split(
            cluster_ids,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
    else:
        # Chaque fold est constitué de clusters complets et stratifiés.
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        if min(labels.count(label) for label in set(labels)) < n_folds:
            raise ValueError("Chaque classe doit contenir au moins n_folds clusters")
        folds = list(splitter.split(cluster_ids, labels))
        train_val_indices, test_indices = folds[fold_index]
        train_val_ids = [cluster_ids[i] for i in train_val_indices]
        test_ids = [cluster_ids[i] for i in test_indices]

    # La validation est extraite du pool train, sans toucher au fold de test.
    if not train_val_ids or not cluster_ids:
        raise ValueError("Aucun cluster disponible pour l’entraînement")
    # val_size est une fraction du jeu complet, même avec un fold de test.
    relative_val_size = val_size * len(cluster_ids) / len(train_val_ids)
    if not 0 < relative_val_size < 1:
        raise ValueError("val_size trop grand pour le pool hors test")
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=relative_val_size,
        random_state=seed,
        stratify=[class_by_cluster[cid] for cid in train_val_ids],
    )

    def keys_for(cluster_subset):
        return [
            key
            for cluster_id in cluster_subset
            for key in keys_by_cluster[cluster_id]
        ]

    return keys_for(train_ids), keys_for(val_ids), keys_for(test_ids)