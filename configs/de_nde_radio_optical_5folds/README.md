# Radio seul et radio + optical — DE/NDE

Les cinq fichiers `fold_0.config` à `fold_4.config` sont des configurations JSON
acceptées par `python -m clusterprep`. Chaque fichier lance successivement :

1. `DualEncoderCNN`, radio uniquement (`modalities: ["radio"]`).
2. `DualEncoderCNN`, radio + LEGACY optique g/r/z, un encodeur par modalité.

Mode multimodal, tâche `de_nde`, MPS avec `torch.compile`/Inductor `default`.
Même protocole pour les deux expériences : 200 époques maximum, patience 50,
batch 16, images 128×128, AdamW à 5e-5, OneCycleLR et MixUp activés.
Augmentation géométrique : rotations et retournements, sans zoom ni bruit ajouté.

Les folds externes sont identiques entre expériences (graine 42) : chaque fold
sert une fois de test ; `val_size: 0.2` définit la validation selon le splitter
existant. Aucune CV interne ni réentraînement supplémentaire : 10 entraînements.
Les modalités manquantes restent autorisées pour conserver la même cohorte :
les bandes optical absentes sont masquées et leur contribution est annulée.
La normalisation optical est ajustée uniquement sur le train de chaque fold.

Sources radio : `scratch/data/PSZ2/classified`.
Sources optical : `scratch/data/PSZ2/fits/<cluster_id>/<cluster_id>_legacy.fits`.
Résultats : `results/de_nde_radio_optical_5folds/fold_<index>`.
Ces chemins sont relatifs à la racine PH_Project_I, fixée par le script.

Depuis PH_Project_I :

```sh
./clusterprep/run_de_nde_radio_optical_5folds.sh --dry-run
./clusterprep/run_de_nde_radio_optical_5folds.sh
# Reprendre uniquement les folds 3 et 4 :
./clusterprep/run_de_nde_radio_optical_5folds.sh 3 4
```

Le script utilise `.venv/bin/python` si disponible ; `PYTHON_BIN` permet de
choisir un autre interpréteur. Il lance les expériences séquentiellement et
s'arrête à la première erreur. Une relance crée de nouveaux runs.
