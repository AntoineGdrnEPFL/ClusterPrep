# Métriques, seuils et notebook RR/RH — compte rendu

Les tâches du fichier `codex_multilabel_analysis_spec.md` ont été implémentées. Le notebook fonctionne sur une démonstration synthétique autonome et sur les deux expériences RR/RH présentes dans `results/experiments`.

## Code livré

- `src/clusterprep/metrics.py` : seuils indépendants avec compatibilité de `threshold=0.5`/`mask`, alias `target_mask`, validation des seules cibles connues, TN/FP/FN/TP, prévalence et supports, balanced accuracy par label, Brier, hamming loss et nombre de labels effectivement agrégés. `classification_metrics` DE/NDE conserve son implémentation.
- `src/clusterprep/training.py` : sélection optionnelle des seuils après restauration du meilleur checkpoint, uniquement sur validation ; test évalué une seule fois ; exports cohérents avec les seuils choisis ; provenance des folds et paramètres scattering enregistrée pour les nouveaux runs. Les comparaisons RR/RH vérifient aussi les CSV et manifests.
- `src/clusterprep/config_wizard.py`, `__init__.py` : nouvelles options et API publiques de seuils.
- `src/clusterprep/multilabel_analysis.py` : chargement et validation des schémas, annotations par cluster, cohortes compatibles, tables d'erreurs, courbes, bootstrap de clusters, résumés de folds et exports protégés.
- `notebooks/analyse_multilabel.ipynb` : six sections d'analyse, sept cellules de code exécutables, configuration initiale portable et mode synthétique explicite.
- `examples/execute_multilabel_notebook.py` : exécution réelle du notebook avec un kernel utilisant le Python du venv, sauvegarde des sorties et refus d'écraser un dossier existant.
- `tests/test_multilabel_analysis.py` : 25 nouveaux tests/cas paramétrés, en complément des 89 tests existants.
- `pyproject.toml`, `README.md`, `notebooks/README_multilabel.md` : extra `analysis`, lancement, contrat des artefacts, conventions et limites.

Les CNN/SSN et leurs forwards n'ont pas été modifiés pour cette tâche. Les fichiers de configuration et notebooks déjà modifiés par l'utilisateur n'ont pas été remplacés. Les expériences réelles sont uniquement lues ; les nouvelles analyses sont dans des dossiers distincts.

## Validation exécutée

Depuis `/Users/antoinegdrn/PH_Project_I` :

```bash
PYTHONPATH=clusterprep/src MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/clusterprep-mpl IPYTHONDIR=/tmp/clusterprep-ipython \
  XDG_CACHE_HOME=/tmp/clusterprep-cache \
  .venv/bin/python -m pytest clusterprep/tests -q
```

**114 tests réussis, 0 échoué, 0 ignoré ; 33 sous-tests réussis, 17,61 secondes.** Les huit avertissements sont sept rappels de la convention d'annotation RR/RH et une dépréciation Matplotlib dans le notebook historique. [Journal complet](../results/multilabel_analysis_pytest.log).

La suite vérifie les valeurs exactes sur les quatre combinaisons binaires, les NaN masqués, les probabilités sigmoid indépendantes, les scores continus AP/AUC, les seuils distincts, les ex æquo déterministes, les contraintes de précision impossibles, les agrégats indéfinis, la sérialisation, les corruptions de manifests/CSV, la comparaison des politiques/vérités, les permutations de lignes et le bootstrap reproductible.

Un entraînement synthétique RR/RH de deux époques avec `candidate_policy=ignore` et sélection F1 a été exécuté dans les tests. Le test d'intégration contrôle les seuils enregistrés dans le checkpoint, le JSON et les prédictions, ainsi que **l'unique appel d'évaluation finale du test**. Modifier les labels et probabilités test n'affecte pas la sélection basée sur validation.

Toutes les cellules du notebook ont aussi été exécutées dans IPython pendant pytest. Deux exécutions supplémentaires ont utilisé un vrai kernel Jupyter :

```bash
PYTHONPATH=clusterprep/src MPLBACKEND=Agg .venv/bin/python \
  clusterprep/examples/execute_multilabel_notebook.py \
  results/multilabel_notebook_validated_20261011 \
  --bootstrap-iterations 50 --threshold-method f1

PYTHONPATH=clusterprep/src MPLBACKEND=Agg .venv/bin/python \
  clusterprep/examples/execute_multilabel_notebook.py \
  results/multilabel_analysis_real_20261011 --bootstrap-iterations 200 \
  --run-dir results/experiments/experience_1-20261011T100616-22b6ac63 \
  --run-dir results/experiments/experience_1-20261011T101358-e3231577
```

Résultats : **7/7 cellules de code exécutées, 0 erreur, code retour 0 pour chaque exécution**. L'essai Jupyter initial était bloqué par les sockets du sandbox ; les exécutions réussies ont eu lieu après autorisation du noyau local. Aucun serveur de données externe n'a été contacté.

`compileall` et `git diff --check` ont également réussi. Les figures PR/ROC et de comparaison ont été inspectées visuellement ; les libellés de modèles ont été ajustés pour rester lisibles.

## Artefacts consultables

- [Notebook source](notebooks/analyse_multilabel.ipynb).
- [Guide de lancement et conventions](notebooks/README_multilabel.md).
- [Notebook exécuté sur données synthétiques](../results/multilabel_notebook_validated_20261011/analyse_multilabel.executed.ipynb).
- [Notebook exécuté sur vos deux expériences](../results/multilabel_analysis_real_20261011/analyse_multilabel.executed.ipynb).
- [Preuve d'exécution synthétique](../results/multilabel_notebook_validated_20261011/execution.json).
- [Preuve d'exécution réelle](../results/multilabel_analysis_real_20261011/execution.json).
- [Tableau comparatif réel](../results/multilabel_analysis_real_20261011/exports/analysis-20261011T110058-f8cd670f/comparison-group-1.csv).
- [Configuration de l'analyse réelle](../results/multilabel_analysis_real_20261011/exports/analysis-20261011T110058-f8cd670f/analysis_config.json).

Les dossiers d'export contiennent les prévalences/combinaisons, métriques par label et globales, matrices de confusion, courbes PR/ROC et d'apprentissage, histogrammes, courbes de seuil sur validation, tables FP/FN, sous-populations paired/raw_only, intervalles bootstrap, figures et provenance des seuils.

## Analyse des expériences existantes

Les deux expériences sont compatibles sur les mêmes **14 clusters test** ; chacune possède 39 clusters train et 14 validation. Les annotations, masques, disponibilités CHANDRA et politiques sont identiques. Les seuils historiques RR=RH=0,5 ont été conservés (`stored`) ; aucun seuil n'a été optimisé sur ces tests.

| Modèle | Macro-F1 | Macro AP | Macro ROC-AUC |
|---|---:|---:|---:|
| CNN multimodal | 0,6727 | 0,8042 | 0,8292 |
| DualEncoderCNN multimodal | 0,4583 | 0,5691 | 0,4583 |

Ces valeurs sont descriptives. Le test contient **RR : 4 positifs / 10 négatifs ; RH : 12 positifs / 2 négatifs**, selon les cibles enregistrées. Ces petits supports rendent notamment les estimations RH et les intervalles instables. Les sorties ne justifient pas une affirmation de supériorité statistique d'une architecture.

## Hypothèses et limites

- La convention historique « annotation absente = négative » est conservée et explicitement signalée ; son bien-fondé astrophysique n'est pas établi par les artefacts. Les masques existants sont respectés ; candidats et confirmés sont distingués uniquement lorsque le manifeste le permet.
- AP et ROC-AUC exigent deux classes connues dans cette implémentation ; les agrégats ignorent les labels non évaluables et en indiquent le nombre. Les valeurs impossibles sont `null` dans les JSON.
- Le bootstrap est conditionnel aux modèles et seuils figés. Il n'intègre pas l'incertitude des annotations ni la sélection des hyperparamètres. Les réplications dégénérées et le nombre de réplications valides sont explicités ; les macro-métriques conservent leur règle d'agrégation sur les labels évaluables par réplication.
- Les anciens résultats réels ne contiennent pas de provenance de fold : aucun résumé inter-fold n'a été inventé. La logique de résumé de folds disjoints et le refus de chevauchements ont été testés sur fixtures.
- `has_chandra` décrit la présence d'un fichier, pas sa couverture spatiale effective. La fraction de couverture n'est pas exportée par ces résultats et reste indisponible dans l'analyse.
- La reliability diagram était optionnelle : Brier et histogrammes sont fournis, sans calibration de probabilités ni fit supplémentaire.
- Le notebook n'entraîne pas les modèles et ne valide pas leur exécution GPU. La démonstration synthétique utilise des scores simulés explicitement identifiés, avec historique d'entraînement vide.

Aucun problème bloquant logiciel connu ne subsiste pour les fonctionnalités demandées. La validation scientifique des négatifs et les faibles effectifs restent des limites d'interprétation, documentées dans le notebook.
