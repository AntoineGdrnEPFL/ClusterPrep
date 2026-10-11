# Audit `clusterprep` — 11 octobre 2026

**La pipeline corrigée fonctionne sur les données synthétiques CPU. Elle n'est pas encore validée pour un entraînement scientifique RR/RH sur le catalogue réel : la définition des négatifs doit être confirmée.** Aucun fichier astrophysique source n'a été modifié et aucun téléchargement réel n'a été lancé.

## A. Bilan

L'audit a porté sur tous les modules de `src/clusterprep`, les tests, le lancement CLI et les outils de préparation du catalogue. L'état initial échouait sur **15 des 52 tests unittest** exécutés. Après correction : **89 tests pytest réussis, 0 échec, 0 ignoré**, avec **33 sous-tests réussis**. Les sept avertissements finaux correspondent à six rappels de convention scientifique RR/RH et une dépréciation Matplotlib dans le notebook d'analyse.

Les fonctionnalités suivantes ont été exécutées : inventaire et déduplication, compatibilité des anciens dictionnaires `class`, cibles exclusives et multi-label, candidats positifs ou masqués, trois modes de Dataset, 24 variantes synchronisées, caches mémoire/FITS/scattering, forwards/backwards des six architectures, entraînement et restauration de checkpoints, métriques, exports, comparaisons, courbes et quatre commandes CLI.

Principaux défauts identifiés et corrigés :

- `splits.py` et la validation d'expériences utilisaient encore `class` alors que l'inventaire produisait `labels` ; d'autres chemins ne reconnaissaient plus les dictionnaires historiques.
- L'inventaire perdait CHANDRA si le fichier se trouvait dans une autre catégorie que le premier RAW ; les conflits entre sources CHANDRA n'étaient pas vérifiés.
- Le précalcul scattering imposait DE/NDE, empêchant les amas seulement RR/RH d'être traités. Il exige maintenant uniquement les images.
- `run_experiments()` utilisait `config` avant son initialisation sans splits explicites. Les ensembles de labels n'étaient pas sérialisables dans le manifeste JSON.
- La tâche RR/RH pouvait exporter des noms de sorties DE/NDE. L'ordre explicite NDE/DE n'était pas respecté partout.
- La valeur `candidate_policy="all"` était ambiguë et transformait les candidats en négatifs. Elle est désormais refusée ; `ignore` dispose d'un masque explicite et testé.
- Les métriques et les pertes n'intégraient pas le masquage des candidats. Les poids positifs incluent maintenant uniquement les négatifs connus du train.
- La moyenne des losses CE pondérées dépendait du découpage en lots. L'agrégation respecte maintenant la normalisation globale de CrossEntropyLoss.
- Le cache mémoire ne se renouvelait pas après modification des sources. Les signatures incluent maintenant ctime, y compris lorsque mtime est restauré.
- Le cache scattering ne vérifiait pas le contenu numérique, le checksum ou les dimensions théoriques. Ces vérifications et la détection de modification des sources pendant le calcul ont été ajoutées.
- Le parseur de catalogue classait `DE` seul comme « unclassified », perdait une confirmation coexistante avec un candidat et dupliquait parfois cDE.
- Le téléchargeur supposait gzip même pour une archive `.tar`, supprimait les données précédentes avant validation et extrayait des chemins non contrôlés. Une extraction temporaire validée conserve désormais l'ancien jeu en cas d'échec.

## B. Modifications

| Fichiers | Changement et justification |
|---|---|
| `labels.py` (nouveau) | Adaptation `class`/`labels`, validation DE/NDE, génération commune des cibles et masques ; limite la duplication métier. |
| `io.py` | Découverte des six catégories présentes, fusion des annotations, vérification des copies RAW et CHANDRA, conservation des clés historiques `*.fits`. |
| `dataset.py` | Cibles et masques communs, cache mémoire invalidable, accès aux images sans cible pour le scattering ; paramètres invalides refusés. |
| `processed.py` | Ajout de ctime à la provenance des FITS ; les anciennes métadonnées déclenchent une reconstruction. |
| `scattering.py` | Précalcul indépendant des labels, 24 variantes, version 2, checksum SHA-256, forme/dtype/valeurs finies, métadonnées temporaires uniques, déduplication par amas. |
| `splits.py` | Stratification selon la tâche, regroupement par amas, repli reproductible avec avertissement pour combinaisons rares. |
| `models.py` | Validation explicite de tous les axes des coefficients précalculés. Les architectures et hyperparamètres sont conservés. |
| `training_core.py` | `pos_weight` calculé sur les effectifs connus par label, vérification des effectifs. |
| `training.py` | Validation des splits, refus des doublons dans un split, splits partagés par tâche/politique, seed, pertes masquées, MixUp masqué, exports JSON/CSV, noms et ordre des sorties, prédiction compatible avec TrainingConfig historique. |
| `metrics.py` | Métriques RR/RH sur annotations connues, scores non définis à `null`, subset accuracy sur les lignes entièrement annotées. |
| `config_wizard.py`, `__main__.py` | Choix task/politique, découverte automatique RR/RH, suites mixtes et comparaison séparée par tâche. Le scattering reste automatiquement associé aux modèles SSN. |
| `__init__.py` | Exports publics de DualEncoderCNN et multilabel_metrics. |
| `visualization.py` | Suppression des impressions de débogage ; courbes effectivement générées. |
| `categorise_PSZ2.py` | Parseur de catégories corrigé et remplacement des liens symboliques cassés. |
| `rsync_PSZ2.py` | Archives tar/gzip, extraction contrôlée en zone temporaire, restauration en cas d'échec, dossier de sortie configurable, absence de création de dossiers à l'import. |
| `pyproject.toml` | Extras `test` et `download`, sans nouvelle dépendance obligatoire de la pipeline. |
| `tests/test_audit.py`, `tests/test_training_protocol.py` | Régressions synthétiques et mise à jour de l'attente mathématique de la loss CE agrégée. |
| `examples/audit_smoke.py`, `README.md` | Reproduction CLI autonome et documentation des conventions et limites. |

`preprocessing.py` et `progress.py` ont été lus et validés sans modification. La normalisation par image utilise uniquement les pixels et le masque, jamais les labels. Le comportement de sélection du premier plan non céleste des cubes est conservé avec son avertissement existant.

## C. Tests et expériences réellement exécutés

Environnement : Python 3.12.14, torch 2.14.1, torchvision 0.29.1, NumPy 2.5.3, Astropy 8.0.1, reproject 0.21.0, scikit-learn 1.9.1, Kymatio 0.3.0, pytest 9.1.1. **MPS et CUDA indisponibles** selon PyTorch ; validation sur CPU uniquement. Pytest a été installé dans le venv du projet après autorisation ; Kymatio était déjà présent.

Depuis `/Users/antoinegdrn/PH_Project_I` :

```bash
# État initial : 52 tests, 15 erreurs
PYTHONPATH=clusterprep/src MPLCONFIGDIR=/tmp/clusterprep-mpl \
  .venv/bin/python -m unittest discover -s clusterprep/tests -q

# État final : 89 réussis, 0 échoué, 0 ignoré, 33 sous-tests réussis
PYTHONPATH=clusterprep/src MPLCONFIGDIR=/tmp/clusterprep-mpl \
  XDG_CACHE_HOME=/tmp/clusterprep-cache MPLBACKEND=Agg \
  .venv/bin/python -m pytest clusterprep/tests -q

# Catalogue synthétique et commandes CLI réelles ; quatre runs de deux époques
PYTHONPATH=clusterprep/src MPLCONFIGDIR=/tmp/clusterprep-mpl \
  XDG_CACHE_HOME=/tmp/clusterprep-cache MPLBACKEND=Agg \
  .venv/bin/python clusterprep/examples/audit_smoke.py \
  results/clusterprep_audit_final_20261011
```

Le script a exécuté `python -m clusterprep --list-models`, `createconfig --output ...` avec réponses par défaut, `preprocess ...`, puis `python -m clusterprep training_config.json`. Les quatre commandes ont renvoyé le code 0. `git diff --check` et la compilation Python ont également été exécutés. Les premiers passages ayant révélé des erreurs ont été corrigés et réexécutés ; aucun test non exécuté n'est compté comme réussi.

Les entraînements persistants utilisent 60 amas synthétiques 16×16, avec les six catégories, RR+RH et présence/absence de CHANDRA. Les modèles sont entraînés en mode multimodal ; les tests couvrent aussi raw et pair.

| Tâche / modèle | Époques exécutées | Meilleur checkpoint | Loss test | Macro-F1 test |
|---|---:|---:|---:|---:|
| DE/NDE — SimpleCNN | 2 | 2 | 0,780146 | 0,428571 |
| RR/RH — SimpleCNN | 2 | 2 | 0,566714 | 0,375000 |
| RR/RH — DualSSN | 2 | 1 | 0,555501 | 0,750000 |
| RR/RH — DualEncoderSSN | 2 | 1 | 0,560590 | 0,000000 |

Ces scores sur bruit synthétique vérifient l'exécution et ne mesurent aucune performance astrophysique. Le test pytest exécute aussi une expérience `ignore` avec augmentation et MixUp, une validation uniquement partiellement connue, l'ordre NDE/DE et des contrôles de reproductibilité historiques.

Chaque run persistant contient `best.pt`, `config.json`, `splits.json`, `history.csv`, `history.jsonl`, `validation_predictions.csv`, `test_predictions.csv`, `metrics.json`, `status.json` et `history.png` ; les SSN incluent `scattering.json`. Les comparaisons sont séparées par tâche. La loss des cohortes est non pondérée, tandis que la loss de sélection utilise le critère d'entraînement : ces deux colonnes ne sont pas interchangeables.

Artefacts et preuves :

- [Rapport pytest](../results/clusterprep_audit_pytest.log).
- [Résumé des runs](../results/clusterprep_audit_final_20261011/summary.json).
- [Vérifications numériques et intersections](../results/clusterprep_audit_final_20261011/verification.json).
- [Configuration de la suite](../results/clusterprep_audit_final_20261011/training_config.json).
- [Journal CLI d'entraînement](../results/clusterprep_audit_final_20261011/cli-3.log).
- [Erreurs de la suite initiale](../results/clusterprep_audit_final_20261011/baseline-unittest.log).

## D. Vérifications scientifiques et mathématiques

**Fuites.** Les intersections train/validation/test sont nulles dans les tests et dans les quatre expériences persistantes. Les 24 variantes sont créées après séparation. Aucun label ne participe au preprocessing ni à l'identité du cache. Précalculer les images des splits ne réalise aucun ajustement statistique global. Les poids se fondent uniquement sur le train. Le test n'intervient ni dans le checkpoint ni dans le seuil fixé à 0,5.

**Labels et losses.** DE/NDE conserve des cibles `long`, deux logits et softmax/CE. RR/RH utilise des cibles float32 `[B,2]`, deux logits et sigmoid/BCE. RR+RH reste un seul amas avec `[1,1]`. Les candidats ignorés n'interviennent ni dans les effectifs négatifs, ni dans la loss, ni dans les métriques. Le test de gradient vérifie explicitement un gradient nul sur une annotation masquée. MixUp transporte la même permutation pour cibles et masques ; il est désactivé pour les SSN et refusé par le moteur s'il y est appliqué.

**Métriques.** Précision, rappel, F1, AUC, AP, matrices 2×2 par label, macro/micro-F1, hamming et subset accuracy sont testés, y compris sans positif, sans négatif, prédictions nulles, double positif et masques vides. Les AUC/AP non définies valent `null`. La subset accuracy utilise uniquement les lignes entièrement connues et vaut `null` si elles sont absentes ; `n_fully_known` et `n_known` explicitent les dénominateurs. Les métriques par label sont exportées dans les CSV d'historique et de comparaison.

**Scattering.** Les tests comparent les 24 variantes en RAW et multimodal au calcul Kymatio direct après la même normalisation et augmentation, avec `atol=2e-6`, `rtol=2e-5`. Une vérification supplémentaire sur les coefficients persistants `[24,3,9,4,4]` mesure une erreur absolue maximale de **0,0** sur CPU. Les tests interdisent explicitement tout appel Kymatio pendant les forwards des SSN et vérifient les gradients des branches. Le premier SSN crée 50 caches ; le second en réutilise 50 sans nouveau calcul, leur configuration d'entrée étant identique.

**Géométrie.** Les augmentations des trois canaux sont comparées aux transformations indexées attendues ; les masques restent binaires. Les tests existants vérifient la reprojection WCS, la réduction, les NaN, le recouvrement nul et l'annulation de la branche CHANDRA des encodeurs séparés. La reprojection bilinéaire d'une image contenant des NaN peut agrandir la zone invalide : il ne faut pas attendre une égalité RAW/CHANDRA pixel à pixel dans ce cas.

## E. Décisions et limites avant données réelles

1. **Priorité scientifique : définir les vrais négatifs RR/RH.** La convention historique « annotation absente = négative » est conservée et signalée, mais n'est pas prouvée par les dossiers. `candidate_policy="ignore"` ne masque que les candidats du label concerné. Il faut confirmer l'exhaustivité des annotations ou fournir un masque d'observation explicite pour toutes les annotations inconnues avant de tirer des conclusions scientifiques.
2. **Définir la population éligible.** Les amas uniquement DE ou NDE restent exclus de RR/RH ; l'inclusion de NDE comme `[0,0]`, la signification de cDE et les correspondances entre labels demandent une décision explicite. Aucun changement de ce type n'a été introduit dans le Dataset.
3. **Valider les données astrophysiques.** L'audit n'a pas exécuté de run sur les vrais FITS : aliases d'identifiants, provenance des annotations, alignement WCS réel, couverture et éventuels cubes spectraux restent à examiner. L'identité par nom FITS ne résout pas deux noms différents pour un même objet physique.
4. **Valider le matériel et l'échelle cible.** MPS/CUDA, entraînements longs à 128×128, charge mémoire du cache, workers multiples et écritures concurrentes de plusieurs processus n'ont pas été validés ici. Les signatures utilisent les métadonnées du système de fichiers, pas un hash de chaque FITS à chaque accès.
5. **Évaluer les splits rares.** Le repli aléatoire est annoncé et reproductible ; il ne garantit pas une stratification multi-label optimale. Le train refuse explicitement un label sans positif ou sans négatif connu. Les petits splits de validation/test peuvent produire des métriques non définies.
6. **Outils réseau.** Le téléchargement a été testé avec des réponses et archives simulées, pas contre le serveur LOFAR ; la stabilité du HTML distant et les téléchargements complets restent non vérifiés. Le parseur refuse les contradictions DE/NDE ; il ne retire pas automatiquement d'anciennes annotations déjà présentes dans un répertoire classifié.

**Conclusion : prête techniquement pour des essais CPU encadrés, avec tests et artefacts reproductibles ; pas encore certifiée pour un entraînement scientifique RR/RH réel tant que les points 1 et 2 ne sont pas tranchés et le catalogue réel contrôlé.**
