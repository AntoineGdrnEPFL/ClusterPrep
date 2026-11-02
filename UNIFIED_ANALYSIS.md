# Analyse unifiée des entraînements

Point d’entrée : [analyse_trainings.ipynb](notebooks/analyse_trainings.ipynb).
Installer l’environnement d’analyse avec `pip install -e '.[analysis]'`, ouvrir le
notebook et renseigner `ROOT_DIR`. L’exécution de haut en bas fonctionne avec un
dossier vide, des expériences partielles ou plusieurs tâches. Aucun poids n’est
chargé et aucun résultat d’entraînement n’est modifié.

## Audit des formats actuels

- `config.json` contient les métadonnées et la configuration imbriquée `config` ;
  `metrics.json` conserve également la configuration, l’ordre des classes, les
  métriques validation/test et les informations de sélection/arrêt.
- `status.json` fournit le statut explicite. Un ancien run sans marqueur est
  `unverified` : ses résultats sont visibles, mais exclus des moyennes de runs
  terminés. Une fin annoncée sans métriques valides devient `incomplete`.
- `history.jsonl` contient `epoch`, `train`, `validation` et éventuellement `phase`,
  `phase_epoch`, `learning_rates`. Le CSV historique reste accepté en secours.
- Prédictions binaires : `key`, `cluster_id`, `label`, `prediction`,
  `probability_<classe>`, `has_chandra`. La décision enregistrée est un argmax.
- Prédictions RR/RH : `label_RR/RH`, `probability_RR/RH`, `known_RR/RH` et
  éventuellement `prediction_RR/RH`. Les masques historiques sont dérivés seulement
  lorsque la politique d’annotation enregistrée le permet.
- `splits.json` conserve les clés et le manifeste `samples`. Les folds usuels sont
  décrits dans `split_config`. La nouvelle CV fournit `execution.stage`,
  `cv/folds.json`, des dossiers `cv/trial_*/fold_*` et un réentraînement `final`.
  Le numéro du fold absent des JSON est déduit de cette structure **uniquement**
  lorsque `execution.stage=fold`, avec un avertissement explicite.
- Les folds internes peuvent ne pas fournir de CSV. Le lecteur utilise alors
  `validation_epochs/epoch_<best_epoch>.json`. Cette sélection correspond aux
  métriques du run (souvent la dernière epoch), et n’est pas présentée comme
  l’optimum global sélectionné par la CV.
- Les phases disposent de `training_phases.json`. Leurs sous-dossiers de métriques
  ne sont pas comptés comme de nouveaux modèles.

Aucune extension des artefacts d’entraînement n’est nécessaire. Les informations
absentes restent inconnues. Une future extension utile serait d’enregistrer
explicitement `fold_index` dans les métadonnées d’exécution de la CV.

## Backend et compatibilité

Le package [analysis](src/clusterprep/analysis/) sépare découverte/normalisation,
comparaisons, graphiques, navigation et exports. Il expose notamment
`discover_runs`, `registry_table`, `epoch_table`, `prediction_table`,
`metrics_table`, `aggregate_metrics`, `paired_comparisons` et
`out_of_fold_predictions`.

L’ancienne logique binaire a été déplacée sans modification dans
[binary.py](src/clusterprep/analysis/binary.py) ;
[training_analysis.py](notebooks/training_analysis.py) conserve les imports
historiques. Les fonctions de métriques et le lecteur/renderer multilabel
existants sont réutilisés. Aucun ancien notebook n’a été supprimé.

## Conventions statistiques

- L’identité d’un run utilise son chemin résolu : les noms `fold_0` récurrents ne
  collisionnent pas. Les alias de chemins ne créent pas de doublons.
- L’identité de configuration conserve hyperparamètres, seed, modalités,
  prétraitement, transfert, phases, protocole de split et empreinte de code
  disponible. Les réglages d’exécution sans incidence analytique sont exclus.
- Les tâches, définitions de labels, masques, politiques de seuil et protocoles
  incompatibles sont séparés. Les métriques sauvegardées et recalculées ne sont
  pas agrégées ensemble. Les pertes sauvegardées conservent leur pondération.
- Les moyennes donnent une voix par fold : répétitions du même fold moyennées
  d’abord, puis moyenne et écart-type échantillonnal. Les colonnes `n_folds`,
  `expected_folds`, `available_folds` et `<metric>_n` montrent la couverture réelle.
  Une seule observation donne un écart-type indéfini, pas zéro.
- Les historiques s’agrègent par configuration, protocole, phase et epoch locale,
  avec une voix par fold. Aucune interpolation ni extrapolation ; `count` donne
  le nombre de contributeurs disponibles. Une moyenne tardive peut donc être
  affectée par la sélection des runs qui continuent à s’entraîner.
- L’appariement vérifie les identifiants des **trois** partitions, les vérités
  évaluées, les masques et les politiques de seuil. Les différences de score
  ne sont pas calculées lorsque l’appariement n’est pas établi.
- La CV interne et le test final restent séparés. L’agrégation OOF exige une
  configuration/protocole unique, une même epoch de sélection, des validations
  disjointes et une couverture complète de l’univers d’entraînement. Des
  prédictions répétées sur un test fixe ne sont jamais assimilées à de l’OOF.
- Aucun seuil implicite 0,5 n’est introduit. Si les seuils/provenances sont absents,
  seuls les scores continus sont recalculés. Une absence de classe rend l’AUC/AP
  indéfinie. Les annotations inconnues sont exclues label par label.
- Les agrégations entre folds reposent sur le plan CV ou la stratégie de split
  enregistrée et le même univers annoté. Elles restent descriptives ; les
  intervalles de bootstrap historiques par amas ne quantifient pas l’incertitude
  de sélection d’architecture ou de protocole.

## Navigation, images et exports

Les filtres de la cellule de configuration s’appliquent à toutes les sections.
`ENABLE_WIDGETS=True` active des sélections multiples si `ipywidgets` est installé
(dépendance optionnelle). Après changement, exécuter « Appliquer » et les cellules
suivantes. `MAX_DETAIL_RUNS` limite les figures ; les tableaux globaux restent
complets. `LEGACY_ADVANCED` conserve les audits d’annotations, combinaisons,
courbes de seuil de validation, bootstrap et diagnostics binaires historiques.

Les galeries sont optionnelles et limitées en taille. `GALLERY_VIEW` sélectionne
`radio`, `overlay` ou `chandra`. Les quatre groupes RR/RH et les groupes TP/TN/FP/FN
utilisent le renderer existant. Il attend des images **déjà prétraitées et alignées**
par la pipeline ; il ne prétend pas corriger le WCS de FITS bruts arbitraires.
L’absence de CHANDRA dans le manifeste conserve la vue radio ; en vue CHANDRA
seule, elle est indiquée explicitement. Les gabarits se règlent en configuration.

Les exports créent un sous-dossier horodaté avec suffixe unique, à l’extérieur des
runs. CSV du registre, métriques, epochs, prédictions et transfert ; PNG classés
par analyse ; rapport JSON avec la sélection, les configurations et problèmes.
Les figures détaillées utilisent l’identité unique du run pour éviter les collisions.

## Validation

Tests synthétiques : découverte à plusieurs profondeurs, racine vide, alias,
corruption, statut incomplet, scores connus binaires et RR/RH, seuils non 0,5,
masques partiels, folds manquants/répétés, phases et longueurs variables, absence
de prédictions/CHANDRA, comparaisons incompatibles, OOF et export protégé.

```bash
python -m pytest tests/test_unified_analysis.py tests/test_training_analysis.py \
  tests/test_multilabel_analysis.py tests/test_metrics.py -q -k 'not test_notebook_smoke'
python examples/execute_unified_analysis.py --root ../results --output /tmp/analysis-check
```

Validation locale : 49 tests ciblés réussis ; exécution complète du nouveau notebook
sur les 230 runs disponibles (150 binaires terminés, 76 multilabel terminés et
4 binaires en cours), avec représentants CV/transfert et exports. Les 8 runs de
transfert sont identifiés. Les figures de diagnostic et les quatre galeries sur un run réel ont été vérifiées ;
une galerie radio avec contours CHANDRA a également été inspectée visuellement.

Le test préexistant `test_multilabel_analysis.py::test_notebook_smoke` échoue dans
l’ancien notebook : sa galerie de démonstration cherche des FITS `synthetic-*`
absents. Ce test est exclu de la commande ciblée ci-dessus ; le nouveau notebook
possède son propre test d’exécution sur dossier vide et une validation sur données
réelles. Aucun résultat de test complet du projet n’est revendiqué.

## Dossier radio_noise : modèle × classification × fold

Le notebook pointe désormais par défaut sur `../results/radio_noise` (chemin
résolu depuis le package). La racine reste configurable.

- `CLASSIFICATION_FILTER = "rr_binary"` ou `"rh_binary"` distingue les cibles,
  tandis que `TASK_FILTER = "binary"` sélectionne la famille de classification.
- `MODEL_FILTER = "fusion"` ou `"dual_encoder_cnn"` sélectionne l’architecture.
- `FOLD_FILTER = [0, 2, 4]` sélectionne plusieurs folds (un entier est aussi accepté).
- `NOISE_FILTER = "gaussian_pixel"` ou `"beam_correlated"` sélectionne le bruit
  déclaré dans `config.augmentation.radio_noise`, sans inférence par nom de dossier.

Les tableaux agrégés et exports affichent explicitement `task` et `noise_method`.
La section 5 présente les métriques par fold et une matrice classification × modèle
× bruit, avec une colonne par fold. Les graphiques séparent RR et RH et conservent
les deux stratégies de bruit comme configurations distinctes.
`MAX_COMPARISON_CONFIGS` limite les configurations affichées en conservant **tous**
leurs folds ; `MAX_DETAIL_RUNS` limite seulement les diagnostics individuels.

Vérification sur ce dossier : 40 runs terminés, 8 configurations (2 modèles ×
2 classifications × 2 types de bruit), chacune avec les folds 0 à 4.

## Noms des runs

Les tableaux du notebook, légendes, titres et galeries utilisent `run_name` :
le nom du dossier, sans le suffixe final `-YYYYMMDDTHHMMSS-xxxxxxxx` de la pipeline.
Par exemple, `gaussian_pixel_dual_encoder_cnn_rr_binary_multimodal_fold_1-20261031T190114-6f1f9211`
devient `gaussian_pixel_dual_encoder_cnn_rr_binary_multimodal_fold_1`.
Un dossier sans ce suffixe garde son nom intégral. `DETAIL_RUN_NAMES` permet de
sélectionner les diagnostics avec ces noms.

Les vues agrégées utilisent le même nom sans le suffixe `_fold_N` correspondant
au fold enregistré. Les identifiants uniques restent dans les données et exports
pour les jointures et pour distinguer les réexécutions ; le nom affiché ne modifie
pas les règles de regroupement. Les dossiers sources ne sont pas renommés.

## Diagnostics binaires et matrices de confusion

Les diagnostics binaires ont une seule rangée de quatre panneaux pour la classe
cible : DE, RR ou RH. Le choix repose sur la tâche et les noms des classes, et non
sur une position fixe dans les sorties. Les diagnostics multilabel gardent deux
rangées, RR et RH.

La confusion utilise la palette legacy `Blues` (blanc à bleu), avec une échelle
fixe de 0 à 100 %. Chaque case affiche `pourcentage% (nombre d’amas)`. Le
pourcentage est normalisé par classe réelle (ligne), comme le rappel par classe
dans le diagnostic legacy. Une classe sans observations affiche `— (0)`.
Les axes portent les noms des classes plutôt que des coordonnées décimales.

Les PNG des diagnostics individuels et galeries commencent par le nom lisible du
run. Un suffixe technique unique évite l’écrasement de deux exécutions portant le
même nom ; les figures de comparaison collective gardent leur nom de synthèse.
