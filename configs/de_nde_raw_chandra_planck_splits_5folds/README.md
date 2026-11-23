# DualEncoderCNN RAW et RAW + Chandra : splits Planck DE/NDE

Les cinq configurations contiennent les listes train/validation/test exactes des
expériences enregistrées dans `results/de_nde_planck_5folds/fold_0` à `fold_4`.
Les deux expériences Planck de chaque fold ont des listes identiques, ordre inclus.
`split_sources` garde les chemins des manifestes d'origine et `split.splits`
fournit directement ces listes à la pipeline : aucun découpage n'est recalculé.

Deux entraînements par fold : RAW, puis RAW + Chandra, tous deux avec
DualEncoderCNN. Hyperparamètres et normalisation repris des configurations
sauvegardées des bras Planck correspondants, en retirant Planck.
Chandra reste optionnel avec masque pour conserver tous les amas.
200 époques maximum, patience 50, seed 42, batch 16, MPS ; test final activé.

Depuis la racine du projet :

```bash
bash clusterprep/run_de_nde_raw_chandra_planck_splits_5folds.sh --dry-run
bash clusterprep/run_de_nde_raw_chandra_planck_splits_5folds.sh
# Ou uniquement certains folds :
bash clusterprep/run_de_nde_raw_chandra_planck_splits_5folds.sh 0 2
```

Les dix runs sont séquentiels. Chaque run crée un sous-dossier unique dans
`results/de_nde_planck_5folds/fold_<index>/`, sans écraser les runs existants.
Un nouveau lancement crée de nouveaux runs et ne reprend pas automatiquement
les entraînements déjà terminés.
