# Analyse RR/RH multi-label

Ouvrir `analyse_multilabel.ipynb` dans Jupyter et exécuter les cellules de haut en bas. Par défaut, le notebook crée des artefacts synthétiques clairement identifiés : aucune donnée utilisateur, aucun FITS et aucun entraînement n'est nécessaire.

Depuis le dossier parent de `clusterprep` :

```bash
.venv/bin/python -m pip install -e './clusterprep[analysis,test]'
.venv/bin/python -m jupyterlab clusterprep/notebooks/analyse_multilabel.ipynb
```

Le kernel doit utiliser le même environnement Python que `clusterprep`. Pour une exécution automatisée qui impose le bon interpréteur :

```bash
PYTHONPATH=clusterprep/src MPLBACKEND=Agg .venv/bin/python \
  clusterprep/examples/execute_multilabel_notebook.py results/nouvelle_analyse \
  --bootstrap-iterations 200
```

Le dossier de sortie doit être nouveau. Ce script exécute réellement un noyau Jupyter et conserve le notebook exécuté, ses sorties, les PNG et les tableaux. Il ne modifie pas les résultats sources. Un environnement qui interdit les sockets locales peut bloquer Jupyter ; les tests proposent aussi une exécution de toutes les cellules dans IPython sans noyau réseau.

## Choisir les expériences

Dans le bloc initial :

```python
RUN_DIRS = [Path("../results/experiments/mon_run")]
RESULTS_ROOT = None
COHORT = "all"             # all, paired ou raw_only
THRESHOLDS = None          # seuils sauvegardés ; ancien schéma : 0.5 / 0.5
THRESHOLD_METHOD = "stored"
SEED = 42
BOOTSTRAP_ITERATIONS = 200
EXPORT_ROOT = Path("analysis_exports")
```

Les chemins sont relatifs au répertoire de lancement du kernel. `RESULTS_ROOT` peut remplacer `RUN_DIRS` pour découvrir récursivement les expériences RR/RH. Une racine absente ou sans résultat est signalée ; elle ne déclenche pas silencieusement la démonstration.

En ligne de commande, répéter `--run-dir` :

```bash
PYTHONPATH=clusterprep/src MPLBACKEND=Agg .venv/bin/python \
  clusterprep/examples/execute_multilabel_notebook.py results/nouvelle_analyse_reelle \
  --run-dir results/experiments/mon_run \
  --run-dir results/experiments/autre_run
```

Chaque notebook crée un sous-dossier `analysis-<date>-<identifiant>`. Les CSV/JSON refusent l'écrasement ; les figures sont également protégées. Le tableau de synthèse est triable via `SORT_BY` et exporté dans `experiments.csv`.

## Contrat d'entrée et compatibilité

Fichiers requis pour chaque run :

- `config.json` : configuration imbriquée sous `config`, ou ancien objet de configuration direct.
- `metrics.json` : tâche `rr_rh` et noms des sorties. Les métriques affichées sont recalculées depuis les scores continus, pas copiées sans contrôle.
- `splits.json` : listes `train`, `validation`, `test` et dictionnaire `samples` contenant `cluster_id` pour chaque clé.
- `validation_predictions.csv`, `test_predictions.csv` : `key`, `cluster_id`, `has_chandra`, `label_RR`, `label_RH`, `probability_RR`, `probability_RH`.

Champs utilisés lorsqu'ils existent :

- `known_RR`, `known_RH` : masques binaires. Sans ces colonnes, un ancien CSV ne peut être interprété que via les annotations et la politique sauvegardée. Une politique `ignore` sans informations suffisantes est refusée.
- `prediction_RR`, `prediction_RH` : vérifiées contre les seuils sauvegardés.
- `history.jsonl` : courbes train/validation ; champs absents ou scores non définis explicitement indiqués.
- `threshold_selection` dans `metrics.json`, `evaluation` dans `config.json` : seuils choisis et provenance.
- `splits.json.split_config` : fold et seed de split enregistrés par les nouveaux `run_experiments`. Les anciens folds restent inconnus ; aucun index n'est déduit du nom du run.
- `scattering_config` ou `model_params` : J, L et max_order lorsque réellement enregistrés.

Le cas historique `class_names=["DE","NDE"]` associé à une tâche explicitement RR/RH et des colonnes RR/RH est reconnu avec avertissement. Une véritable expérience DE/NDE est refusée par ce notebook ; son notebook historique reste indépendant.

Les clusters sont uniques dans chaque split et les intersections doivent être vides. Le lecteur vérifie la correspondance CSV/manifeste, les annotations et les masques, les probabilités finies dans [0,1] et la disponibilité CHANDRA. Les deux probabilités ne sont jamais soumises à une contrainte de somme égale à 1. `has_chandra` indique la présence d’un fichier associé, pas le pourcentage de couverture spatiale ; ce dernier n’est pas disponible dans les CSV actuels et n’est pas inventé.

## Métriques et dénominateurs

`multilabel_metrics` conserve `threshold=0.5` et `mask=` et accepte désormais :

```python
multilabel_metrics(y, p,
    thresholds={"RR": 0.4, "RH": 0.65},
    target_mask=known)
```

Ne pas fournir simultanément `mask` et `target_mask`, ni un `threshold` non défaut avec `thresholds`. Les seuils doivent être strictement entre 0 et 1. Les valeurs cibles inconnues peuvent être NaN ou -1 si leur masque vaut zéro ; seules les cibles connues doivent être 0/1. Les scores probabilistes restent tous finis.

Par label : précision, rappel, F1, TN/FP/FN/TP, matrice 2×2, ROC-AUC, `average_precision`, prévalence, Brier, balanced accuracy, `n_positive`, `n_negative`, `n_known`. La baseline PR est la prévalence des positifs parmi les cibles connues du label.

ROC-AUC, AP et balanced accuracy exigent ici deux classes connues. Elles valent `None`/`null` autrement, y compris AP avec uniquement des positifs. Précision/rappel/F1 utilisent `zero_division=0` lorsqu'il existe des observations connues ; sans observation connue, ils valent `None`.

Les macro-moyennes ignorent les valeurs indéfinies et publient `macro_f1_n_labels`, `roc_auc_macro_n_labels`, `average_precision_macro_n_labels`, `balanced_accuracy_n_labels`, `brier_macro_n_labels`. La hamming accuracy porte sur toutes les **cellules connues** (`n_known`), la subset accuracy sur les **lignes entièrement connues** (`n_fully_known`). Les métriques impossibles sont sérialisées à `null`, jamais à NaN JSON.

Les CSV utilisent les noms effectifs `RR_average_precision`, `RH_average_precision`, `RR_f1`, `RH_f1`, etc., sans alias ambigu `RR_ap`. `support` reste un alias rétrocompatible de `n_positive`.

## Sélection des seuils sans fuite

Pour les nouveaux entraînements, dans l'objet d'expérience JSON :

```json
{
  "task": "rr_rh",
  "thresholds": {"RR": 0.5, "RH": 0.5},
  "threshold_selection": "f1",
  "threshold_grid": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
  "threshold_min_precision": 0.8
}
```

`threshold_selection="fixed"` est le défaut et conserve le comportement historique. `f1` maximise le F1 de chaque label ; `recall_at_precision` maximise le rappel sous la précision minimale demandée. La grille par défaut va de 0.05 à 0.95 par pas de 0.05. Les ex æquo favorisent le seuil le plus proche de 0.5, puis le plus élevé. En absence de deux classes connues ou de seuil satisfaisant la contrainte, le seuil initial reste appliqué et la raison est enregistrée.

`run_experiment` restaure d'abord le meilleur checkpoint, produit ses prédictions validation, choisit les seuils sur cette validation, puis évalue **une fois** le test. Les seuils sont enregistrés dans `thresholds.json`, `metrics.json`, `config.json.evaluation` et `best.pt.evaluation`. L'historique et `config.json.config` décrivent les seuils utilisés pendant l'entraînement ; les champs `evaluation` décrivent ceux à appliquer au checkpoint restauré.

Dans le notebook, le défaut `stored` réutilise cette provenance. Pour une étude définie **avant d'examiner le test**, `THRESHOLD_METHOD="f1"` ou `"recall_at_precision"` sélectionne sur les CSV validation uniquement. `fixed` applique `THRESHOLDS`. Tous les graphiques/test/cohortes suivants réutilisent ensuite les mêmes seuils figés. Changer ces paramètres après avoir inspecté le test constituerait une sélection exploratoire, pas une estimation finale indépendante.

## Comparaisons, bootstrap et folds

Les comparaisons directes exigent les mêmes identifiants test, vérités connues, masques, couverture et politique de candidats. Les modèles sont comparés dans des groupes séparés quand ces critères diffèrent. Aucune intersection de test n'est fabriquée silencieusement pour rendre les runs comparables. Le chargeur vérifie aussi les manifests avant `compare_experiments` pour les nouveaux appels RR/RH.

Les IC percentile à 95 % rééchantillonnent les clusters uniques, à seuils figés, avec seed fixe. Les CSV indiquent `n_valid` : les réplications où une métrique est indéfinie sont exclues pour cette métrique. Les macro-métriques conservent leur règle d'agrégation des labels évaluables à chaque réplication. Aucun réajustement de seuil, calibration ou sélection du meilleur modèle n'est effectué dans le bootstrap.

Ces IC sont descriptifs et conditionnels au modèle/seuil retenu ; ils ne quantifient pas l'incertitude d'annotation ou de sélection. Peu de positifs/négatifs rend les estimations fragiles. Les folds documentés et disjoints sont résumés par moyenne/écart-type, sans assimiler les entraînements aux folds chevauchants à des expériences indépendantes.

## Limites scientifiques conservées

La convention existante « annotation absente = négative » reste signalée comme **hypothèse à valider**, non comme preuve de négatif fiable. La politique `positive` inclut les candidats ; `ignore` les masque si le même label n'est pas confirmé. Le notebook distingue candidats/confirmés uniquement lorsque `splits.json.samples.labels` le permet. Les CSV restent la vérité évaluée ; aucune annotation n'est réinterprétée en regardant les prédictions.

La reliability diagram optionnelle n'est pas ajoutée ; le Brier score et les histogrammes sont fournis. Les seuils ne calibrent pas les probabilités. L'analyse ne relance pas les CNN/SSN et ne modifie pas leurs forwards.
