# Phase 7B — pipeline historique avec matched_resolution

Aucun entraînement lancé pendant l'intégration. Les cinq fichiers `fold_0.json` à `fold_4.json` et leurs partitions n'ont pas été modifiés.

## Exécution depuis la racine

Vérification seule :

```sh
.venv/bin/python clusterprep/run_matched_5folds.py --device auto
```

Dix entraînements séquentiels : RAW folds 0–4, puis T50kpcSUB folds 0–4 :

```sh
.venv/bin/python clusterprep/run_matched_5folds.py --run --device auto --seed 42 --max-epochs 200 --output outputs/phase7b_unified
```

Relancer la même commande pour continuer la série. Les résultats complets ne sont ignorés que si plan, configuration et budget correspondent. Une tentative interrompue repart de zéro dans un nouveau dossier ; aucune reprise à l'époque interrompue n'est annoncée. Une seule instance du script à la fois.

## Chemin partagé

`run_matched_5folds.py` → `matched_runner.run` → `training.run_experiment` → `training._run` → `ClusterDataset`, `build_model`, DataLoaders historiques, `Trainer.fit`.

Le runner apparié ne contient plus de boucle d'entraînement, de modèle construit directement, de loss, de checkpoint ou de calibration propres. Les anciennes commandes `matched_runner evaluate/report` et le mode `budgeted` ne font plus partie du CLI unifié ; les résultats pilotes déjà produits restent intacts.

Configurations effectives : [RAW](../matched_historical/RAW.json) et [T50kpcSUB](../matched_historical/T50kpcSUB.json). Elles reprennent les hyperparamètres DualEncoderCNN du JSON historique : batch 16, 200 époques maximum, AdamW LR 5e-5 / weight decay 0,1, OneCycle, MixUp, patience 10, seuils fixes 0,5, `augment=true`, `drop_last_train=true`. Le modèle n'est pas modifié et aucun transfert n'est activé.

Les 24 variantes géométriques historiques sont synchronisées entre radio, CHANDRA et masque. Folds 0–1 : 41 amas × 24 = 984 exemples, 61 batches complets ; folds 2–4 : 42 × 24 = 1008 exemples, 63 batches. Le dernier batch incomplet est abandonné conformément au JSON historique. `label_smoothing=0.1` est conservé mais la BCE RR/RH historique ne l'applique pas.

## Différences nécessaires

- Prétraitement physique matched_resolution et cache apparié inchangés, masque radio commun.
- Statistiques globales P30/P99 + asinh(alpha=10) réajustées sur le train non augmenté de chaque fold, séparément pour chaque radio, puis figées en validation. Les anciens fichiers de bornes servent à valider la provenance du plan ; leurs valeurs ne sont pas injectées dans l'entraînement historique.
- Métadonnées physiques et `rr_rh_known_mask` conservées depuis le manifeste. Aucune annotation source changée.
- Partitions explicites communes, pas de nouveau splitting.
- `evaluate_test=false` : le test réservé est retiré des datasets, des signatures de sources, du prétraitement et de la normalisation dès l'entrée dans `_run`. Les identifiants restent consignés dans `splits.json` sous `reserved_external_test`. Aucun fichier de prédictions test produit.

Les 13 amas réservés sont exactement ceux du test fold 0 de phase 6, non utilisés par les pilotes. Les 52 autres amas fournissent train/validation des cinq folds. CHANDRA est présent pour toute cette cohorte. Le modèle conserve son comportement historique face à une couverture nulle.

## Résultats et compatibilité

Chaque tentative est dans `outputs/phase7b_unified/{A|B}_foldN_seed42/attempt_NNN/`. `request.json` contient la demande et `completed.json` pointe vers le sous-dossier créé par le pipeline historique. Le log stdout/stderr est à côté, dans `attempt_NNN.log`.

Exports historiques : `config.json`, `splits.json`, `normalization.json`, `history.csv`, `history.jsonl`, `best.pt`, `thresholds.json`, `metrics.json`, `status.json`. La branche sans test ajoute `validation_metrics.json`, `validation_predictions.json` et `validation_predictions.csv`. Le format des checkpoints et historiques est celui de `Trainer.fit`, pas celui du précédent runner pilote.

## Tests sans entraînement

```sh
.venv/bin/python -m pytest clusterprep/tests/test_matched_runner.py clusterprep/tests/test_matched_full_epochs.py clusterprep/tests/test_phase7b_script.py clusterprep/tests/test_matched_historical_integration.py clusterprep/tests/test_matched_dataset.py -q
```

Tests d'orchestration avec appels simulés, intégration historique avec `fit` et `evaluate` simulés, vérification des 24 transformations et des labels, partitions communes, normalisation train uniquement, exclusion du test, cache et non-régression du dataset historique. Aucun optimizer step ni entraînement n'est exécuté par cette sélection.

Résultat de la vérification : **51 tests réussis en 15,23 s**. Deux avertissements attendus dans le petit jeu synthétique signalent des ROC-AUC indéfinies (classe unique / label inconnu). Les cinq partitions ont aussi été comparées aux manifestes et les configurations RAW/T50 vérifiées identiques hors nom et produit radio. Le dry-run des dix commandes a réussi.
