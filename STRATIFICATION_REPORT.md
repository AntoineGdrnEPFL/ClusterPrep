# Stratification RR/RH — 11 octobre 2026

## Changements

- `src/clusterprep/labels.py` : règles communes, NDE négatif confirmé,
  refus NDE + DE/RR/RH/candidat, regroupement des annotations par amas.
- `src/clusterprep/splits.py` : compatibilité des annotations `class` et `labels`,
  quatre strates RR/RH, candidats ignorés représentés par `?`, regroupement
  avant sélection, stratification des deux séparations et des folds.
  Les strates rares déclenchent la stratification itérative des états connus
  positifs/négatifs et inconnus. Erreurs avec effectifs si impossible ou si
  la dépendance manque. Aucun repli aléatoire RR/RH.
- `src/clusterprep/dataset.py` : cibles issues du même helper que les splits ;
  union des annotations des alias. Images, augmentations et caches conservés.
- `src/clusterprep/training.py` : splits adaptés aux tâches et à la population
  pair RR/RH, déduplication des splits automatiques, diagnostic JSON par run.
  Raw et multimodal de même tâche/politique partagent leurs splits ; pair
  RR/RH a sa population propre. Les contrôles existants refusent les comparaisons
  de cohortes différentes. Les splits explicites restent validés strictement.
- `pyproject.toml` : ajout de `iterative-stratification>=0.1.9` (0.1.9 installé).
- `README.md` : conventions et fonctionnement actualisés.
- `tests/test_stratification.py` : 15 nouveaux tests paramétrés/régressions.
- `tests/test_audit.py` : catalogue synthétique corrigé (anciennement NDE+RH
  contradictoires) et validation du JSON produit par les entraînements réels.
- `tests/test_multilabel_analysis.py` : isolation du smoke test des chemins
  personnels renseignés dans le notebook, sans changer celui-ci.

## Validation

Commande depuis `/Users/antoinegdrn/PH_Project_I` :

```sh
PYTHONPATH=clusterprep/src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/clusterprep-mpl \
IPYTHONDIR=/tmp/clusterprep-ipython XDG_CACHE_HOME=/tmp/clusterprep-cache \
.venv/bin/python -m pytest clusterprep/tests -q
```

**Résultat final : 129 tests réussis, 33 sous-tests réussis, aucun échec, en 17,31 s.**
Les cinq avertissements correspondent à quatre ROC-AUC volontairement
indéfinies dans un test de candidats masqués et à une dépréciation Matplotlib
préexistante.

Le journal complet est dans `../results/stratification_pytest.log`.
Les tests couvrent le découpage DE/NDE exact par comparaison à sklearn,
les quatre combinaisons, les cinq folds, les alias, la reproductibilité,
les contradictions, les candidats masqués, la dépendance indisponible,
les populations pair, les suites mixtes et des entraînements synthétiques
sur deux epochs avec écriture effective des diagnostics.

## Distribution exécutée

`../results/stratification_example.json` provient de 80 amas synthétiques,
20 de chaque strate, avec cinq folds, fold 0, seed 17, validation 20 %.

| Split | 00 | 10 | 01 | 11 | Total |
|---|---:|---:|---:|---:|---:|
| train | 12 | 12 | 12 | 12 | 48 |
| validation | 4 | 4 | 4 | 4 | 16 |
| test | 4 | 4 | 4 | 4 | 16 |

## Limites explicites

Les proportions itératives sont approximatives ; des strates très rares ne
peuvent apparaître dans tous les folds. Les diagnostics indiquent les positifs,
négatifs et inconnus de chaque label et avertissent si la ROC-AUC est indéfinie.
Un entraînement sans positifs/négatifs connus pour chaque label reste refusé.
`ignore` conserve les exemples totalement masqués (`??`), sans contribution
supervisée ; ils ne sont pas comptés comme négatifs connus.

Le catalogue chargé doit contenir NDE : une sélection explicite de dossiers
limitée à RR/RH ne peut pas fournir ces négatifs. Utiliser `class_names: null`
(découverte automatique) ou inclure NDE dans la sélection d'inventaire.
La tâche DE/NDE conserve sa sélection explicite DE/NDE et ses séparations
historiques ; aucune annotation DE n'est ajoutée implicitement aux anciens
catalogues RR seuls. Les anciens résultats ne sont pas réécrits.

Les runs pair RR/RH peuvent avoir un autre test que raw/multimodal ; pour une
comparaison directe, fournir une population commune et des splits explicites.
Les modifications existantes des configurations et notebooks utilisateur sont
conservées. Aucun entraînement long sur les données scientifiques réelles
n'a été lancé pour cette validation.
