# DualEncoderCNN RAW + Planck — RR/RH, 236 amas

Les cinq configurations reprennent exactement, ordre inclus, les splits des runs `dual_encoder_radio_rr_rh_fold_*` de [la suite RAW](../../../results/rr_rh_for_different_modalities_multi_encoder_cnn/). Chaque configuration contient les listes explicites, le chemin source et son SHA-256. Les splits sont aussi identiques à ceux de [la suite DE/NDE RAW + Planck](../de_nde_raw_planck_raw_splits_5folds/).

| Fold | Train | Validation | Test |
|---|---:|---:|---:|
| 0 | 140 | 48 | 48 |
| 1 | 141 | 48 | 47 |
| 2 | 141 | 48 | 47 |
| 3 | 141 | 48 | 47 |
| 4 | 141 | 48 | 47 |

Classification multilabel `rr_rh` : RR et RH sont deux sorties indépendantes. `class_names: null` découvre toutes les catégories d’annotation ; le pipeline utilise ensuite les sorties RR/RH. Les étiquettes et masques suivent le pipeline historique, avec `candidate_policy: positive` (cRR positif RR, cRH positif RH) et NDE négatif pour les deux sorties. Aucune conversion des annotations en DE/NDE. Seuils fixes de 0,5 pour RR et RH.

Modèle `dual_encoder_cnn`, modalités `radio` et `planck`, RAW, prétraitement `historical`, images 128×128, MPS, 200 époques maximum, patience 50, batch 16. Même paramétrage que la suite DE/NDE RAW + Planck. Normalisation radio asinh P30/P99 et Planck standard, ajustées sur le train de chaque fold, sans bornes pré-calculées ni transfert de poids. Test évalué après chaque entraînement.

Depuis la racine du projet :

```bash
bash clusterprep/run_rr_rh_raw_planck_raw_splits_5folds.sh --dry-run
bash clusterprep/run_rr_rh_raw_planck_raw_splits_5folds.sh
```

Ajouter un numéro (0 à 4) pour sélectionner un fold. Les folds sont exécutés séquentiellement. `PYTHON_BIN` permet de remplacer `.venv/bin/python`. Résultats : `results/rr_rh_raw_planck_raw_splits_5folds/fold_<i>/`.

Le dry-run valide les paramètres uniquement. Les projections et la couverture Planck des images sont traitées au chargement des données ; aucun entraînement n’est lancé lors de la génération des configurations.
