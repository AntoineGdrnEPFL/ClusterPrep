# RR et RH binaires : fusion et dual_encoder_cnn sur cinq folds

Les fichiers [fold_0.json](fold_0.json), [fold_1.json](fold_1.json),
[fold_2.json](fold_2.json), [fold_3.json](fold_3.json) et
[fold_4.json](fold_4.json) contiennent chacun les quatre expériences :

| Tâche | Modèles |
| --- | --- |
| rr_binary | fusion, dual_encoder_cnn |
| rh_binary | fusion, dual_encoder_cnn |

## Une commande

Depuis `PH_Project_I` :

```sh
bash clusterprep/run_binary_subtypes_5folds.sh
```

Le [script](../../run_binary_subtypes_5folds.sh) lance séquentiellement les cinq
configurations, soit **20 entraînements**, et s'arrête si une commande échoue.
Il se place automatiquement dans le répertoire du projet et utilise son
`.venv/bin/python`, avec possibilité de choisir un autre interpréteur via
`PYTHON_BIN`. Aucun entraînement n'est lancé lors de la préparation des fichiers.

Afficher les commandes sans entraîner :

```sh
bash clusterprep/run_binary_subtypes_5folds.sh --dry-run
```

Lancer uniquement certains folds, par exemple les folds 2 à 4 :

```sh
bash clusterprep/run_binary_subtypes_5folds.sh 2 3 4
```

Relancer un fold crée de nouveaux runs ; cela ne reprend pas automatiquement un
checkpoint interrompu.

## Protocole

- Mode `multimodal` : radio et CHANDRA ; les amas sans CHANDRA restent inclus avec
  le masque de modalité existant.
- `candidate_policy: positive`, soit cRR assimilé à RR et cRH à RH.
- Cinq folds **de test** : `split.n_folds: 5`, indices 0 à 4 et seed 42 fixe.
  La validation représente 20 % du catalogue complet : environ 60 % train,
  20 % validation et 20 % test par fold, avec arrondis de stratification.
- Les modèles partagent les mêmes splits. Avec cette politique et le catalogue
  actuel, les tâches RR et RH partagent aussi ces splits. Les ensembles de test
  des cinq folds sont disjoints.
- `cross_validation: null` : ces configurations utilisent les cinq folds du
  découpage externe, sans ajouter une CV interne et un réentraînement final.
- 200 époques maximum, batch 16, learning rate `5e-5`, weight decay `0.1`.
- Early stopping sur `val_loss`, patience 50 ; ordonnanceur OneCycle,
  augmentations, MixUp et pondération des classes activés selon les réglages
  actuels de la pipeline.
- Images 128 × 128, `width: 16`, périphérique `mps`. Pour une autre machine,
  modifier `device` dans les expériences (`cpu` ou `cuda`).

Les dossiers d'annotations sont découverts automatiquement. Les amas DE seuls,
cDE seuls et U sont exclus par les règles binaires. Les sorties sont
`[non_RR, RR]` ou `[non_RH, RH]`, avec une cible scalaire.

## Fichiers produits

Les sorties se trouvent sous
`results/binary_subtypes_multimodal_5folds/fold_N/`, dans un sous-dossier unique
par expérience : checkpoints, métriques, historique, splits et prédictions.
Les comparaisons des deux architectures sont produites séparément pour chaque
tâche, sur tous les amas de test (`comparison_cohort: all`). Les caches de
prétraitement sont partagés sous `processed/`.

Les chemins JSON sont relatifs au dossier de lancement ; le script fixe ce
dossier à `PH_Project_I`. Ces fichiers constituent une suite dédiée et ne
remplacent pas les configurations RR/RH individuelles existantes.
