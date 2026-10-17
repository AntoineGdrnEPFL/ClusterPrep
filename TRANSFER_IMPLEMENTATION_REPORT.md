# Rapport d'implémentation — 11 octobre 2026

Le code du transfert multi-phase est intégré à la pipeline existante.
**Validation exécutée après rétablissement des permissions : 164 tests réussis,
1 cas ignoré et 33 sous-tests réussis. Le pilote court sur données réelles s’est
également terminé sur MPS. Aucun correctif supplémentaire n’a été nécessaire.**

## Fichiers

Modifiés :

- [models.py](src/clusterprep/models.py) : groupes X-ray omis lorsqu'absents en raw ; architectures et clés des paramètres conservées.
- [transfer.py](src/clusterprep/transfer.py), déjà présent : configuration, validation des phases, compatibilité source, chargement strict, traçabilité, conversion de l'ancien ImageCNN.
- [experiment_loader.py](src/clusterprep/experiment_loader.py), déjà présent : audit des expositions source/cible par cluster_id ; résolution existante conservée.
- [training.py](src/clusterprep/training.py) : correction des imports/configuration inachevés, conversion JSON, entraînement commun via fit(), phases, splits, artefacts, cohortes.
- [training_core.py](src/clusterprep/training_core.py) : OneCycleLR respecte les différents learning rates des groupes.
- [config_wizard.py](src/clusterprep/config_wizard.py) : choix du transfert, source, mappings, phases et validation.
- [__init__.py](src/clusterprep/__init__.py) : exports publics du transfert et du résolveur.
- [__main__.py](src/clusterprep/__main__.py) : comparaisons séparées par population de test dans les suites.
- [visualization.py](src/clusterprep/visualization.py) : transitions de phases sur les courbes existantes.
- [README.md](README.md) : liens vers le guide et les exemples.

Créés :

- [test_transfer.py](tests/test_transfer.py) : 14 fonctions de tests pytest, paramétrées pour les architectures, modalités et formats de checkpoints.
- [transfer_image_cnn.json](examples/transfer_image_cnn.json) : configuration complète pointant vers une source réelle terminée.
- [transfer_pilot.py](examples/transfer_pilot.py) : trois expériences comparables et analyse des sous-populations via les API existantes.
- [TRANSFER_LEARNING.md](TRANSFER_LEARNING.md) : interfaces, choix, limites et commandes complètes.
- Le présent rapport.

Les modifications préexistantes aux groupes des modèles ont été conservées.
Le fichier exploratoire `src/clusterprep/test_exp_loader.py` n'a pas été modifié.

## Interfaces et choix

- `TransferConfig` est défini dans transfer.py, importé dans training.py et exporté par clusterprep.
- `ExperimentConfig.transfer` accepte None, un TransferConfig ou un dictionnaire JSON.
- `Trainer.fit_transfer(train_loader, val_loader, *, run_dir=None, phases=None)` appelle la même méthode fit() que les entraînements classiques.
- `Trainer.fit()` accepte désormais les arguments optionnels run_dir et phase pour partager checkpoints et historiques avec run_experiment().
- Les outils de transfert et resolve_transfer_splits sont exportés au niveau du package.
- La validation stricte précède le chargement ; la tête source n'est pas transférée. Les nouveaux groupes doivent apprendre dès la première phase.
- Les couches de fusion déjà incluses dans classifier restent dans head, sans changement des clés de checkpoints.
- Chaque phase recrée AdamW, OneCycleLR le cas échéant et l'early stopping, puis restaure son meilleur état.
- Le modèle final est sélectionné entre les meilleurs checkpoints de phases sur validation, jamais sur test.
- Les BatchNorm gelées restent en eval ; les BCE/sigmoid et métriques multi-label existantes sont conservées.
- L'audit refuse les amas du train source dans la validation/test cibles. Une présence de validation source dans le test cible produit un avertissement séparé.

## Vérifications réellement effectuées

Les commandes suivantes ont été exécutées avec succès après rétablissement des
permissions (PyTorch 2.14.1, Python de l'environnement du projet) :

```sh
PYTHONPATH=clusterprep/src MPLBACKEND=Agg .venv/bin/python -m pytest clusterprep/tests/test_transfer.py clusterprep/tests/test_training.py clusterprep/tests/test_training_protocol.py clusterprep/tests/test_models.py clusterprep/tests/test_config_wizard.py -q
PYTHONPATH=clusterprep/src MPLBACKEND=Agg .venv/bin/python -m pytest clusterprep/tests -q
```

| Suite | Réussis | Échecs | Ignorés | Sous-tests réussis | Durée |
|---|---:|---:|---:|---:|---:|
| Transfert et régressions ciblées | 61 | 0 | 1 | 33 | 7,48 s |
| Suite complète | 164 | 0 | 1 | 33 | 20,06 s |

Ces deux suites se recouvrent : les nombres ne doivent pas être additionnés.
Le cas ignoré est volontaire : FusionCNN ne supporte pas le mode radio seul.
Les architectures de scattering ont été testées, avec Kymatio disponible.

Cinq avertissements ont été émis dans la suite complète : quatre ROC-AUC
indéfinies dans un test synthétique sans positifs, et une dépréciation à venir
de Matplotlib dans le notebook d'analyse. Aucun avertissement n'est un échec.

Les contrôles syntaxiques précédents (24 fichiers Python), la lecture du JSON
et `git diff --check` passent également. Le premier essai avait été bloqué
avant le lancement de Python ; cette limitation de permissions est levée.

## Pilote exécuté sur MPS

Source : [ImageCNN radio DE/NDE terminé](../results/clusterprep/image_cnn/image_cnn_raw-20261003T192326-659482b7).
Le statut enregistré de cette source est completed. Son checkpoint utilise
l'architecture actuelle ImageEncoder.

Contrôle d'inventaire par lecture JSON et noms des FITS, sans exécution du modèle :

| Population | Nombre |
|---|---:|
| Train historique | 124 |
| Validation historique | 42 |
| Test historique | 41 |
| Total historique | 207 |
| Nouveaux candidats cRR/cRH | 29 |
| Catalogue actuel | 236 |
| Amas historiques introuvables | 0 |
| Amas historiques uniquement DE | 0 |

Commande exécutée avec code de sortie 0 :

```sh
PYTHONPATH=clusterprep/src MPLBACKEND=Agg .venv/bin/python clusterprep/examples/transfer_pilot.py --head-epochs 2 --finetune-epochs 2 --no-augment
```

[Résultats complets du pilote](../results/transfer_image_cnn/pilot-20261011T203224-0832f546/),
[comparaison CSV](../results/transfer_image_cnn/pilot-20261011T203224-0832f546/comparison.csv) et
[métriques par sous-population](../results/transfer_image_cnn/pilot-20261011T203224-0832f546/pilot_metrics.json).

Les trois expériences partagent exactement les mêmes **141/48/47 amas** en
train/validation/test. Les 236 FITS prétraités ont été réutilisés. Le bras
scratch a effectué 4 epochs ; la tête seule 2 ; le transfert complet 2 + 2.
Les seuils sont sélectionnés sur validation, et les checkpoints sont choisis
sur val_loss. Les trois statuts sont completed ; les artefacts attendus,
la disjonction des splits, leur identité entre runs et les empreintes des
checkpoints source ont été vérifiés.

Scores sur les 47 amas du test :

| Expérience | AP RR | F1 RR | AP RH | F1 RH | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Scratch | 0,3401 | 0,2593 | 0,4559 | 0,5079 | 0,3836 |
| Tête seule | 0,2523 | 0,3000 | 0,6586 | 0,6000 | 0,4500 |
| Deux phases, modèle sélectionné | 0,2523 | 0,3000 | 0,6586 | 0,6000 | 0,4500 |

La dernière ligne réutilise volontairement le checkpoint classifier_only :
sa meilleure val_loss (0,9602) est inférieure à celle du fine-tuning (0,9684).
Les poids fine-tunés et leurs métriques de validation restent disponibles dans
phases/full_finetuning. Leur test n'a pas servi à la sélection et n'a pas été
évalué séparément.

Sur les 19 amas à émission diffuse du test :

| Expérience | AP RR | F1 RR | AP RH | F1 RH | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Scratch | 0,6949 | 0,5385 | 0,8412 | 0,9143 | 0,7264 |
| Tête seule / modèle retenu | 0,2812 | 0,3000 | 0,8587 | 0,6667 | 0,4833 |

L'amélioration globale de F1 ne s'accompagne donc pas d'une amélioration sur
cette sous-population. Ce pilote très court valide le fonctionnement, sans
établir un gain scientifique : le budget complet et plusieurs graines restent
à étudier. Le test contient 41 amas historiques et 6 nouveaux candidats ; leurs
scores séparés figurent dans pilot_metrics.json.

## Limites restantes

Le résolveur historique exige candidate_policy=positive et un catalogue
canonique à une entrée par cluster_id. Les avertissements d'absence de
métadonnées de normalisation et de changements de fichiers source sont
conservés dans la provenance. Aucune nouvelle dépendance n'a été ajoutée.
Le pilote MPS et la suite de tests passent, mais le budget complet de 15 + 40
epochs et une évaluation multi-graines n'ont pas été exécutés.

La configuration complète et les commandes sont dans le
[guide de transfert](TRANSFER_LEARNING.md).
