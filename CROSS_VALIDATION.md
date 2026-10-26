# Validation croisée interne et réentraînement final

La section `cross_validation` se place dans **chaque expérience** de la liste
`experiments`. La commande existante reste identique :

```sh
PYTHONPATH=clusterprep/src .venv/bin/python -m clusterprep clusterprep/configs/cross_validation/de_nde_multimodal.json
```

Cette commande s'exécute depuis la racine du projet parent. Avec le package installé,
`python -m clusterprep chemin/config.json` convient aussi.

## Configuration monophasée

```json
"cross_validation": {
  "enabled": true,
  "n_splits": 5,
  "shuffle": true,
  "seed": 42,
  "strategy": "auto",
  "selection_metric": "val_loss",
  "selection_mode": "min",
  "max_epochs": 50,
  "retrain_full_train": true,
  "evaluate_test_after_retrain": true,
  "source_provenance_policy": "error"
}
```

`task`, `model`, `mode`, les hyperparamètres, l'initialisation `transfer` et les
`training_phases` gardent leur emplacement habituel. Aucun champ existant ne doit
être migré. Sans cette section, ou avec `enabled: false`, l'entraînement habituel
reste actif. L'assistant `createconfig` accepte cette section comme objet JSON.

`max_epochs` est le budget commun de recherche ; s'il est absent ou `null`, le
champ `epochs` de l'expérience est utilisé. Chaque fold exécute ce budget complet,
sans arrêt anticipé ni restauration d'un meilleur état en cours de trajectoire.
La moyenne des pertes de validation est pondérée par le nombre d'amas de chaque
validation (sans augmentation). Le minimum choisit l'époque, **numérotée à partir
de 1** ; en cas d'égalité, l'époque la plus petite gagne. La dispersion exportée
est l'écart-type pondéré entre folds, pas un intervalle de confiance.

Pour choisir selon F1 macro, utiliser `selection_metric: "val_macro_f1"` (alias
`"f1_macro"`) et `selection_mode: "max"`. Les seuils sont fixés aux valeurs de
`thresholds` pendant toute la recherche. En RR/RH, si `threshold_selection` demande
`f1` ou `recall_at_precision`, les seuils finaux sont ensuite choisis sur les
prédictions hors fold regroupées **à l'époque/combinaison retenue**. Le test n'est
jamais consulté. Le modèle final utilise ces seuils comme valeurs fixes.

## Frontière externe et folds

Le split externe est d'abord résolu par les mécanismes existants : `split`,
`split.splits`, folds externes, ou réutilisation des splits du transfert historique.
Les partitions train et validation historiques sont réunies en un pool de train
externe ; **aucun amas test n'y entre**. La configuration `split.use_val: false`
est également acceptée. Les listes originales et les listes effectives sont
sauvegardées. En mode `pair`, seuls les amas avec Chandra sont éligibles ; les
exclusions sont listées et la frontière train/test est conservée.

Les folds sont construits sur les identifiants uniques d'amas, avant les 24
variantes d'augmentation. Les doublons d'IDs et recouvrements sont refusés.
Chaque amas éligible du train externe apparaît exactement une fois en validation.

- DE/NDE : `StratifiedKFold` sur la classe binaire existante.
- RR/RH : `MultilabelStratifiedKFold` sur les deux indicateurs `[RR, RH]` des amas
  entièrement annotés. La politique `candidate_policy: "positive"` conserve son
  interprétation habituelle des candidats. Avec `"ignore"`, les amas dont une
  cible est inconnue sont répartis séparément par taille et graine ; leur masque
  est conservé. Aucun inconnu n'est codé comme négatif pour la stratification.
  Cette répartition partielle est explicitement signalée et enregistrée.

Les effectifs RR+, RH+, RR+RH+, RR−RH−, négatifs connus, inconnus et proportions
sont sauvegardés pour le pool et chaque partition de chaque fold. Des avertissements
signalent les positifs insuffisants et les AUC non définies. Un sous-train sans
classe/positif/négatif requis est refusé ; diminuer `n_splits`, par exemple de 5 à 3.
Une métrique de sélection non définie provoque un échec explicite, jamais une
agrégation en ignorant silencieusement un fold.

## Phases et transfert

Les phases restent décrites par `training_phases`, ou `transfer.phases` pour le
transfert historique. Leurs noms, groupes de paramètres, LR, weight decay et gels
sont ceux du moteur existant. Pour comparer des durées couplées :

```json
"cross_validation": {
  "enabled": true,
  "n_splits": 3,
  "selection_metric": "val_loss",
  "selection_mode": "min",
  "phase_durations": [
    {"head": 5, "full": 15},
    {"head": 10, "full": 15},
    {"head": 10, "full": 30}
  ],
  "retrain_full_train": true,
  "evaluate_test_after_retrain": true
}
```

Les noms doivent correspondre **exactement** aux phases existantes. Chaque essai
repart du même modèle initial/checkpoint source, sur les mêmes folds. Les phases
s'enchaînent depuis leur **dernier état**, sans restauration du minimum individuel.
La métrique pondérée à la fin de la trajectoire complète classe les combinaisons.
En cas d'égalité : moins d'époques au total, puis durées dans l'ordre des phases,
puis index de l'essai. Sans `phase_durations`, les durées existantes forment un
unique candidat ; aucune durée n'est alors optimisée. `max_epochs` est réservé au
cas monophasé et refusé avec un plan de phases.

La provenance des sources est contrôlée, y compris les transferts antérieurs
référencés par leurs configurations. Les empreintes des poids, configurations
sources et manifestes sont vérifiées avant/après chaque entraînement. Par défaut,
un préentraînement ayant vu les validations internes, ou une provenance absente,
bloque la recherche. Pour une exploration explicitement non certifiée, la politique
`source_provenance_policy: "warn"` permet ces cas et les signale dans les rapports.
Un chevauchement **connu avec le test externe est toujours refusé**, même en mode
`warn`. Pour une CV rigoureuse, fournir des sources préentraînées hors des amas
évalués ; le préentraînement n'est pas automatiquement refait dans chaque fold.

## Statistiques, réinitialisation et réentraînement

Tous les modèles enregistrés et leurs modalités passent par `run_experiment` et
`Trainer` : CNN, fusion, dual encoders, scattering, fusion spatiale et constructeurs
personnalisés compatibles avec l'API existante. Aucun optimiseur parallèle n'est
implémenté pour la CV. Les optimizers, schedulers et états d'arrêt sont nouveaux
pour chaque essai/fold et pour le modèle final. Les graines modèle sont conservées
pour comparer les mêmes initialisations ; la graine CV contrôle les folds.

La normalisation actuelle est déterministe **image par image** (percentiles
30–99, asinh alpha 10), sans statistique ajustée sur une population. Les caches
FITS et scattering déterministes peuvent donc être partagés. Les poids de classes
sont calculés exclusivement sur le sous-train, puis recalculés sur tout le train
final. Les statistiques de BatchNorm appartiennent au nouveau modèle et sont mises
à jour en entraînement uniquement, sauf modules explicitement gelés. Il n'existe
actuellement aucun autre transformateur appris de dataset à ajuster. Un ajout
futur de preprocessing appris devra respecter cette séparation et ne pas introduire
de statistiques de population dans les caches partagés.

Le modèle final repart de zéro ou des mêmes sources de transfert, et exécute
exactement les durées sélectionnées sur **100 % du train externe éligible**. Le
scheduler est recréé avec ce budget final ; une trajectoire OneCycle plus courte
peut donc différer du préfixe d'une trajectoire de recherche plus longue.
Le checkpoint final est sauvegardé puis rechargé avant l'unique passage sur le
test. Les métriques par cohortes proviennent de ces mêmes prédictions, sans nouvelle
inférence. `best.pt` garde son nom pour la compatibilité, mais désigne cet état
final (`selection: "last_epoch"`), pas un checkpoint d'un fold.

La répétabilité des partitions et des graines est testée ; elle ne garantit pas
une identité bit à bit sur tous les accélérateurs et versions de bibliothèques.

## Artefacts et échecs

```text
nom-cv-<date>-<id>/
  config_resolved.json
  environment.json
  external_split.json
  status.json
  cv/
    folds.json
    initialization_audit.json
    trial_0/fold_0/
      config.json, splits.json, split_distribution.json
      history.csv, history.jsonl, best.pt, metrics.json, status.json
      validation_epochs/epoch_1.json, ...
      phases/...                       # si multiphase
    search_progress.json
    cv_summary.json
    epoch_selection.json
    oof_predictions.json
    thresholds.json                   # RR/RH
    cv_curves.png
  final/
    config.json, splits.json, split_distribution.json
    best.pt, history.csv, history.jsonl
    metrics.json, test_predictions.csv, validation_predictions.csv
    thresholds.json                   # RR/RH
    training_phases.json, phases/...   # si multiphase
    status.json
```

La CLI renvoie `final/` comme run de référence et y applique les exports/comparaisons
existants. Les courbes et le chargeur d'analyse RR/RH acceptent la validation finale
vide. Les versions et empreintes des sources Python sont archivées. Le checkpoint
final contient aussi la sélection CV, les sources initiales et le manifeste final.

`retrain_full_train: false` s'arrête après la recherche et n'évalue jamais le test,
même si `evaluate_test_after_retrain` reste à `true`. Avec réentraînement mais
`evaluate_test_after_retrain: false`, le checkpoint final est produit sans métriques
test. La CLI n'inclut pas ces runs dans les tableaux comparant les tests.

Une erreur marque le run et le fold concerné comme échoués ; les historiques déjà
produits restent inspectables. Il n'y a pas de reprise automatique partielle : une
nouvelle commande crée un nouveau dossier et refait la recherche. Un résumé de
sélection n'est produit qu'après tous les folds/essais réussis ; le statut racine
fait foi si le réentraînement final échoue ensuite.

## Exemples et vérification

- [DE/NDE multimodal](configs/cross_validation/de_nde_multimodal.json) : reprend les
  listes historiques du run `10302900`, réunit ses 124+42 amas en train externe et
  conserve ses 41 amas test. Recherche sur 50 époques, 5 folds, puis réentraînement.
- [RR/RH multimodal](configs/cross_validation/rr_rh_multimodal.json) : split externe
  avec 20 % de test, CV interne 5 folds, seuils fixés à 0.5.

Ces exemples lancent de vrais entraînements, potentiellement longs. Les tests
utilisent exclusivement des images synthétiques et de petits budgets :

```sh
PYTHONPATH=clusterprep/src .venv/bin/python -m pytest clusterprep/tests/test_cross_validation.py -q
```
