# DE/NDE : RAW + Planck et RAW + CHANDRA + Planck

Cinq JSON, avec deux expériences par fold :

1. `DualEncoderCNN` : RAW + Planck.
2. `MultiEncoderCNN` : RAW + CHANDRA + Planck.

Tâche `de_nde`, classes dans l’ordre `[DE, NDE]`, prétraitement `historical`, mode `multimodal` : les amas sans CHANDRA sont conservés avec leur masque. Le test est évalué après chaque entraînement (`evaluate_test=true`), sur le checkpoint sélectionné sur la validation. Les statistiques de normalisation proviennent uniquement du train.

Les hyperparamètres viennent des expériences Planck historiques A/C : 200 époques maximum, patience 50, batch 16, learning rate 5e-5, augmentations activées et MPS. Les chemins de sortie et les noms sont spécifiques à cette suite.

Le découpage reprend le protocole DE/NDE existant : `n_folds=5`, `fold_index=0..4`, `split_seed=42`, `val_size=0.2`, inventaire `scratch/data/PSZ2/classified` limité aux classes DE/NDE explicites. Les deux expériences d’un JSON partagent les splits via `run_experiments`. Ce protocole n’utilise pas les partitions RR/RH de phase7b, qui contiennent des objets sans étiquette binaire explicite ; aucune étiquette DE/NDE n’est inférée. La population et les membres exacts des splits dépendent de l’inventaire présent au lancement.

Depuis la racine du workspace :

```sh
# Validation des cinq JSON, sans entraînement
bash clusterprep/run_de_nde_planck_5folds.sh --dry-run

# Dix entraînements séquentiels, deux par fold
bash clusterprep/run_de_nde_planck_5folds.sh

# Seulement certains folds
bash clusterprep/run_de_nde_planck_5folds.sh 0 2
```

`PYTHON_BIN` permet de changer l’interpréteur. Le script se place automatiquement à la racine du workspace et s’arrête en cas d’erreur.

Résultats : `results/de_nde_planck_5folds/fold_<index>/`, avec un sous-répertoire unique par expérience. Les métriques test et prédictions suivent le format habituel de la pipeline.

Validation réalisée : chargement des dix `ExperimentConfig`, vérification des paramètres et des modèles, syntaxe bash et CLI `--dry-run` pour les cinq fichiers. Aucun entraînement lancé lors de la génération.
