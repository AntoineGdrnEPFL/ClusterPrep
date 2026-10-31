# Classification binaire RR et RH

Les tâches `rr_binary` et `rh_binary` réutilisent le dataset, les modèles,
l'entraînement, les métriques et les checkpoints de `de_nde` : cible scalaire
`torch.long`, deux logits, softmax et `CrossEntropyLoss`. L'ordre des sorties est
`[non_RR, RR]` ou `[non_RH, RH]` ; la classe positive a toujours l'indice 1.
`de_nde` conserve son ordre historique et `rr_rh` ses deux cibles multilabel.

## Sélection et cibles

| Annotations | rr_binary | rh_binary |
| --- | --- | --- |
| RR | 1 | 0 |
| RH | 0 | 1 |
| NDE | 0 | 0 |
| RR + RH | 1 | 1 |
| DE seul, cDE seul, U seul | exclu | exclu |
| cRR (`positive`) | 1 | 0 |
| cRH (`positive`) | 0 | 1 |
| RR + cRH (`ignore`) | 1 | exclu |
| RH + cRR (`ignore`) | exclu | 1 |

Avec `ignore`, un candidat de la classe recherchée exclut l'amas, sauf si une
annotation certaine de cette même classe existe. Un candidat seul, sans RR, RH
ou NDE certain, est exclu des deux tâches. Une annotation certaine opposée ne
résout pas l'incertitude : RR+cRH ne devient pas un négatif RH. RR+cRR reste
positif RR et négatif RH. Les contradictions NDE/émission et class/labels lèvent
une erreur. Les annotations des alias d'un même amas sont fusionnées avant
sélection. La sélection et la cible utilisent la même fonction dans
[labels.py](src/clusterprep/labels.py).

## Lancement

Depuis le répertoire parent du dépôt, avec l'environnement Python du projet :

```sh
python -m clusterprep clusterprep/configs/rr_binary.json
python -m clusterprep clusterprep/configs/rh_binary.json
```

Les exemples [RR](configs/rr_binary.json) et [RH](configs/rh_binary.json) utilisent
le schéma JSON existant. On peut aussi changer `task` dans une configuration
binaire existante, conserver ses paramètres de modèle et choisir
`candidate_policy: "ignore"`. Le CLI découvre les dossiers d'annotations présents
si `class_names` est omis ou reste l'ancienne liste DE/NDE. Une liste explicite
plus détaillée conserve son rôle de sélection des dossiers d'entrée ; ce ne sont
pas les noms des deux sorties du modèle. Pour les appels Python, fournir un
catalogue contenant les annotations RR/RH/NDE d'origine.

La pondération des classes conserve les options et valeurs par défaut existantes
(`use_class_weights`, `class_weight`) ; aucune nouvelle stratégie n'est activée.
La sélection de seuils demeure réservée à `rr_rh`.

## Splits et caches

Les splits sont construits par identifiant d'amas avant augmentation. Les
catégories RR, RH, NDE et RR+RH sont préférées pour la stratification, ce qui donne
les mêmes splits RR/RH à graine et population identiques. Avec des catégories
rares, la stratification revient à la cible binaire ; si elle est impossible,
une erreur demande d'adapter les effectifs ou les folds. Avec `ignore`, les deux
populations peuvent différer. Les splits fournis sont contrôlés pour
l'éligibilité et les doublons/fuites d'identifiants. Les classes absentes sont
signalées ; l'entraînement exige les deux classes, et les métriques conservent
`null` pour une AUC non définie.

Les caches de splits distinguent tâche et politique des candidats. Les caches
FITS/scattering restent partagés : ils ne contiennent pas de cibles. Les cibles
sont recalculées selon la tâche et la politique du dataset.

## Résultats et analyse

Les CSV de prédictions conservent identifiant, cible scalaire, prédiction et
probabilités, avec une colonne `annotations` supplémentaire pour ces tâches.
Les métriques indiquent `task`, `positive_class` et `negative_categories`
(effectif et faux positifs par combinaison d'annotations négatives).

[training_analysis.py](notebooks/training_analysis.py) lit ces sorties avec les
outils binaires existants : matrices de confusion, ROC, distributions et
comparaisons. Les scores de gain utilisent `f1_positive`, `recall_positive` et
`precision_positive` pour RR/RH ; les clés DE historiques sont conservées.
`plot_binary_galleries(run, processed_root=...)` produit TP/TN/FP/FN avec le rendu
radio et contours CHANDRA existant. `chandra_only=True` affiche CHANDRA seule.
Adapter les templates de chemins à l'emplacement des FITS déjà prétraités.

## Transfer learning

Utiliser le mécanisme modulaire existant `transfer.sources` / `transfer.mappings`
pour copier les encodeurs DE/NDE et conserver une tête initialisée pour la cible.
Les phases `training_phases` permettent l'entraînement de la tête puis le
fine-tuning. Le format de transfert historique, spécialisé multilabel, reste
réservé à `rr_rh`.

Pour chaque source, l'audit cherche `splits.json` à côté du checkpoint. Un amas
cible de validation/test présent dans le train source provoque une erreur. Un
test cible présent dans la validation source déclenche un avertissement ;
l'évaluation n'est pas déclarée indépendante. Si les splits source manquent,
l'indépendance est non vérifiable et un avertissement est enregistré. Les audits
sont sauvegardés dans `transfer_splits_report.json` et `transfer.json`.

## Fichiers et validation

- Sélection/cibles : [labels.py](src/clusterprep/labels.py),
  [dataset.py](src/clusterprep/dataset.py), [splits.py](src/clusterprep/splits.py).
- Pipeline et configuration : [training.py](src/clusterprep/training.py),
  [training_core.py](src/clusterprep/training_core.py),
  [cross_validation.py](src/clusterprep/cross_validation.py),
  [models.py](src/clusterprep/models.py), [__main__.py](src/clusterprep/__main__.py),
  [config_wizard.py](src/clusterprep/config_wizard.py),
  [experiment_loader.py](src/clusterprep/experiment_loader.py).
- Analyse : [training_analysis.py](notebooks/training_analysis.py),
  [multilabel_analysis.py](src/clusterprep/multilabel_analysis.py).
- Tests : [test_binary_subtypes.py](tests/test_binary_subtypes.py), couvrant les
  candidats, contradictions, splits et CV, cibles scalaires, entraînement court
  dans les trois modes, métriques, checkpoints, transfert des encodeurs,
  configurations et galeries. Les suites historiques couvrent la non-régression.

Les entraînements de test sont synthétiques et ne constituent pas une mesure de
performance astrophysique.
