# DualEncoderCNN multimodal RR/RH — matched_resolution

Dix configurations d'entraînement effectivement lues par le runner :

- `RAW/fold_0.json` à `RAW/fold_4.json` : RAW + CHANDRA.
- `T50kpcSUB/fold_0.json` à `T50kpcSUB/fold_4.json` : T50kpcSUB + CHANDRA.

Chaque JSON est une suite CLI contenant `experiments`, `output_dir` et `matched_cohort`. La section `experiments` contient les paramètres ExperimentConfig, notamment le prétraitement physique, les 24 augmentations historiques, batch 16, 200 époques, patience 10, OneCycle, MixUp, seuils fixes 0,5, normalisation globale train uniquement et `evaluate_test=false`.

Les partitions restent dans `../phase7b_multimodal/fold_N.json`, communes aux deux radios. Le runner charge le manifeste physique et les masques de supervision associés ; il ne régénère pas les splits. Le CLI `python -m clusterprep` reconnaît `matched_cohort`, charge le manifeste physique et les annotations masquées, puis transmet les partitions explicites à la pipeline historique. Le chemin `matched_cohort.source` est relatif au fichier JSON. Le `output_dir` est relatif au répertoire courant, comme pour les suites historiques.

Depuis la racine du projet :

```sh
# Vérifier les dix configurations sans entraîner
bash clusterprep/run_matched_multimodal_5folds.sh --dry-run

# Lancer les dix entraînements séquentiellement
bash clusterprep/run_matched_multimodal_5folds.sh
```

Options : `--device auto|cpu|mps`, `--seed 42`, `--max-epochs 200`, `--output outputs/phase7b_multimodal`. Les valeurs CLI du device, de la seed et des époques remplacent les valeurs correspondantes des JSON. Le nom du run et le cache physique sont déterminés par le fold et le manifeste. Les autres paramètres proviennent du JSON de chaque fold.

Les résultats et journaux sont séparés par expérience, fold et tentative. Relancer la même commande ignore les résultats complets compatibles. Une tentative interrompue repart de zéro dans un nouveau dossier. Arrêt au premier échec, sans lancement des folds suivants.

Les fichiers `RAW.json` et `T50kpcSUB.json` à la racine de ce dossier sont les modèles de configuration antérieurs ; les fichiers par fold ont maintenant priorité. Aucune architecture, partition ou observation n'a été modifiée pour créer ces configurations.

## Lancement direct avec le CLI historique

Depuis la racine :

```sh
.venv/bin/python -m clusterprep clusterprep/configs/matched_historical/RAW/fold_0.json
.venv/bin/python -m clusterprep clusterprep/configs/matched_historical/T50kpcSUB/fold_0.json
```

Ajouter `--dry-run` pour valider sans entraîner. Remplacer `fold_0` par `fold_1` à `fold_4` pour les autres folds. Le lancement direct respecte les hyperparamètres du JSON et crée un nouveau run historique à chaque invocation, dans `output_dir`. Le script séquentiel existant conserve sa gestion des tentatives et des résultats déjà terminés.

La population active est désormais multimodale complète : 154 amas (65 avec CHANDRA, 89 sans), avec 13 test réservés. Voir [le rapport de population](../phase7b_multimodal/README.md). Les anciens folds appariés ne sont plus la source des dix JSON.
