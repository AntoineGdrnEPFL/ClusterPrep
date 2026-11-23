# RR/RH — radio seul

Les cinq fichiers `fold_0.json` à `fold_4.json` lancent chacun une expérience
`DualEncoderCNN`, en mode multimodal, pour la tâche multilabel `rr_rh`.
Cette série contient donc cinq entraînements.

Protocole repris de `../de_nde_radio_optical_5folds` : 200 époques maximum, patience 50,
batch 16, images 128×128, AdamW à 5e-5, OneCycleLR et MixUp activés.
Exécution sur MPS avec `torch.compile`/Inductor `default`.
Rotations et retournements activés, sans zoom ni bruit ajouté.

Cinq folds externes, graine 42, `val_size: 0.2`, sans CV interne ni
réentraînement supplémentaire. Les trois séries RR/RH utilisent les mêmes
paramètres de partitionnement et autorisent les modalités manquantes.
`class_names: null` permet la découverte automatique des catégories originales.
Les candidats sont traités comme positifs (`candidate_policy: positive`) et les
seuils RR/RH sont fixes à 0,5.

Radio : `scratch/data/PSZ2/classified`.
Les modalités supplémentaires suivent les paramètres de la configuration source ;
LEGACY utilise `scratch/data/PSZ2/fits`, avec normalisation ajustée sur le train.
Résultats : `results/rr_rh_radio_5folds/fold_<index>`.

Depuis la racine du projet :

```sh
./clusterprep/run_rr_rh_radio_5folds.sh --dry-run
./clusterprep/run_rr_rh_radio_5folds.sh
# Seulement les folds 3 et 4 :
./clusterprep/run_rr_rh_radio_5folds.sh 3 4
```

Le mode `--dry-run` du script affiche les commandes sans entraîner.
Le script utilise `.venv/bin/python` si disponible ; `PYTHON_BIN` permet de
choisir un autre interpréteur. Les folds sont exécutés séquentiellement,
avec arrêt à la première erreur. Une relance crée de nouveaux runs.
