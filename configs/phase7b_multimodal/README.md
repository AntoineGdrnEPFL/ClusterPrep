# Cohorte multimodale complète — modes indépendants du prétraitement

La sélection CHANDRA dépend exclusivement du mode : `multimodal` conserve tous les amas radio éligibles ; `pair` conserve ceux avec CHANDRA. Cette règle est commune aux prétraitements historique et matched_resolution, avec ou sans liste explicite de modalités. Une ancienne politique `missing_modality_policy=required` ne rend plus X-ray obligatoire en mode multimodal.

## Population et partitions

Sur le même inventaire validé de phase 6, pour RAW comme T50kpcSUB :

| Mode | Historique | matched_resolution | Sans CHANDRA conservés |
|---|---:|---:|---:|
| multimodal | 154 | 154 | 89 |
| pair | 65 | 65 | 0 |

Ces nombres concernent l'inventaire commun validé, pas tous les fichiers bruts du catalogue. Les exigences physiques de l'harmonisation (métadonnées, deux produits radio pour définir le beam commun, WCS exploitable) sont inchangées. Une incompatibilité physique produit une erreur explicite ; elle ne doit jamais être assimilée à une exclusion pour absence de CHANDRA. Le traitement sans CHANDRA existait déjà et reste inchangé : matching radio, image X-ray et masque nuls, statut absent explicite.

Les anciens folds appariés restent dans `../phase7b/` pour la traçabilité des expériences précédentes. Les cinq nouveaux `fold_N.json` sont construits sur les 154 amas : 13 amas du test externe initial sont réservés, les 141 autres participent à la CV. Fold 0 : 112 train / 29 validation ; folds 1–4 : 113 train / 28 validation. Les 89 amas sans CHANDRA participent donc aussi à la validation, au lieu d'être ajoutés uniquement au train. Les combinaisons de labels connus/inconnus sont stratifiées avec seed 42.

Les deux radios utilisent exactement les mêmes partitions. Les bornes de normalisation sont recalculées uniquement sur le train non augmenté à l'exécution, puis figées. Aucun fichier de bornes des anciens folds n'est réutilisé. Le test externe n'entre dans aucun dataset de la CV ; il reste composé d'amas avec CHANDRA, donc sa population d'évaluation finale ne représente pas les amas sans X-ray.

## Exécution depuis la racine

```sh
# Validation seulement
bash clusterprep/run_matched_multimodal_5folds.sh --dry-run

# Dix entraînements séquentiels
bash clusterprep/run_matched_multimodal_5folds.sh

# Un fold directement
.venv/bin/python -m clusterprep clusterprep/configs/matched_historical/RAW/fold_0.json
.venv/bin/python -m clusterprep clusterprep/configs/matched_historical/T50kpcSUB/fold_0.json
```

Les JSON pointent vers ce manifeste. Le script sauvegarde ses tentatives dans `outputs/phase7b_multimodal`, et le CLI direct dans `outputs/phase7b_multimodal_cli`. Les anciennes sorties sont préservées. Avec 24 variantes, batch 16 et drop_last historique, les entraînements ont 168 ou 169 batches par époque.

## Vérification sans entraînement

73 tests réussis : quatre combinaisons mode/prétraitement, CHANDRA absent (chargement et forward du modèle en mode évaluation, aucun backward), masquage effectif des features, augmentations synchronisées, partitions, CLI et intégration historique simulée. Deux avertissements attendus concernent les AUC indéfinies dans une fixture synthétique. Construction des datasets sur les 154 vrais identifiants vérifiée pour les deux produits et les quatre combinaisons ; effectifs consignés dans `population_audit.json`.

Commande :

```sh
.venv/bin/python -m pytest clusterprep/tests/test_mode_semantics.py clusterprep/tests/test_matched_runner.py clusterprep/tests/test_matched_config_cli.py clusterprep/tests/test_phase7b_script.py clusterprep/tests/test_matched_historical_integration.py clusterprep/tests/test_matched_dataset.py -q
```

Aucun entraînement lancé pendant ces corrections.

## Évaluation finale des checkpoints sauvegardés

```sh
# Vérifie uniquement checkpoints présents, provenance, partitions et bornes figées
.venv/bin/python -m clusterprep evaluate --runs-root outputs/phase7b_multimodal_cli --dry-run

# Évalue chaque checkpoint, sans entraînement ni recalibration
.venv/bin/python -m clusterprep evaluate --runs-root outputs/phase7b_multimodal_cli --device auto --output-root outputs/phase7b_external_test
```

Pour un seul modèle, remplacer `--runs-root` par `--run-dir` et le chemin du répertoire contenant `best.pt`. L'évaluation écrit un sous-dossier par modèle avec `test_metrics.json`, `test_predictions.csv`, `test_predictions.json` et `evaluation.json` (provenance, seuils et bornes utilisées). Elle refuse d'écraser une sortie existante. Tous les contrôles préalables sont réalisés avant d'évaluer le premier modèle.

Le checkpoint, les seuils sauvegardés et les statistiques de normalisation train sont figés. Aucune sélection ni calibration n'utilise le test. Les dix modèles voient les mêmes 13 amas réservés : ce ne sont pas 130 observations indépendantes. Ne pas sélectionner le meilleur fold à partir de ces scores test. Le test est petit et composé uniquement d'amas avec CHANDRA ; il ne mesure pas directement la performance en absence de X-ray. Les métriques ignorent les annotations inconnues et rapportent les dénominateurs effectifs.

Validation de la commande : dry-run réussi pour les dix modèles, trois tests automatisés réussis (pas d'accès aux images test pendant le dry-run, fuite rejetée, exports avec loader/évaluation simulés). Aucune évaluation réelle du test externe n'a été exécutée lors de l'ajout de cette commande.
