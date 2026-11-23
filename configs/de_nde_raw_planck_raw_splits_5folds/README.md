# DualEncoderCNN RAW + Planck — DE/NDE, 236 amas

Les cinq configurations reprennent exactement, ordre inclus, les splits sauvegardés des runs `dual_encoder_radio_rr_rh_fold_*` de [la suite RAW](../../../results/rr_rh_for_different_modalities_multi_encoder_cnn/). Les chemins et empreintes SHA-256 des sources sont enregistrés dans chaque configuration. Aucun split n’est régénéré selon les étiquettes DE/NDE.

| Fold | Train | Validation | Test |
|---|---:|---:|---:|
| 0 | 140 | 48 | 48 |
| 1 | 141 | 48 | 47 |
| 2 | 141 | 48 | 47 |
| 3 | 141 | 48 | 47 |
| 4 | 141 | 48 | 47 |

Les 236 amas comprennent 67 DE confirmés, 29 candidats cDE et 140 NDE. `data_class_mapping` inclut explicitement les candidats dans DE (96 DE / 140 NDE), conformément à la politique positive des runs sources. Le chargement conserve leurs annotations d’origine dans `source_labels`. Cette option nécessite la version du CLI livrée avec ces configurations.

Modèle `dual_encoder_cnn`, modalités `radio` et `planck`, RAW et prétraitement `historical`, images 128×128, MPS, 200 époques maximum, patience 50, batch 16. Les hyperparamètres proviennent de la configuration RAW+Planck DE/NDE existante. Normalisation radio asinh P30/P99 et Planck standard, ajustées sur le train de chaque fold ; aucune borne d’un ancien run n’est réutilisée. Aucun transfert de poids. Test évalué après chaque run.

Depuis la racine du projet :

```bash
bash clusterprep/run_de_nde_raw_planck_raw_splits_5folds.sh --dry-run
bash clusterprep/run_de_nde_raw_planck_raw_splits_5folds.sh
```

Pour un seul fold, ajouter son numéro (0 à 4). Le script lance les folds séquentiellement. `PYTHON_BIN` permet de remplacer `.venv/bin/python`.

Les résultats sont écrits dans `results/de_nde_raw_planck_raw_splits_5folds/fold_<i>/`. Le dry-run valide les paramètres ; il ne valide pas les projections Planck et ne lance aucun entraînement. La génération/validation des images et du cache Planck intervient pendant le chargement des données d’entraînement.
