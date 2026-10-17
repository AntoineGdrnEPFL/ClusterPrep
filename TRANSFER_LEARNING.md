# Transfert DE/NDE → RR/RH

Le transfert passe par les mêmes `ExperimentConfig`, `run_experiment`,
`run_experiments`, `Trainer.fit`, datasets, pertes et métriques que les
entraînements classiques. `transfer=None` conserve l'entraînement de zéro.

## Lancer une expérience

Depuis le répertoire `PH_Project_I`, avec l'environnement du projet :

```sh
PYTHONPATH=clusterprep/src .venv/bin/python -m clusterprep clusterprep/examples/transfer_image_cnn.json
```

L'[exemple JSON complet](examples/transfer_image_cnn.json) pointe vers une
expérience ImageCNN radio DE/NDE terminée du projet. Adapter `source_experiment`
et `data_dir` sur une autre installation. Les chemins relatifs sont résolus
depuis le répertoire de travail, comme pour les suites existantes.

La configuration essentielle est :

```json
{
  "name": "ImageCNN_RR_RH_Transfer",
  "task": "rr_rh",
  "model": "image_cnn",
  "mode": "raw",
  "size": 128,
  "candidate_policy": "positive",
  "threshold_selection": "f1",
  "device": "mps",
  "use_scheduler": false,
  "use_mixup": false,
  "transfer": {
    "source_experiment": "results/clusterprep/image_cnn/image_cnn_raw-20261003T192326-659482b7",
    "reuse_splits": true,
    "add_new_clusters": true,
    "allow_subset": false,
    "seed": 42,
    "module_prefixes": {"radio": "encoder"},
    "phases": [
      {
        "name": "classifier_only",
        "epochs": 15,
        "learning_rates": {"head": 0.001},
        "weight_decay": 0.001,
        "patience": 8
      },
      {
        "name": "full_finetuning",
        "epochs": 40,
        "learning_rates": {"radio": 0.00001, "head": 0.0001},
        "weight_decay": 0.001,
        "patience": 12
      }
    ]
  }
}
```

Les identifiants de modèles sont ceux du registre : `cnn`, `image_cnn`,
`fusion`, `dual_ssn`, `dual_encoder_cnn`, `dual_encoder_ssn`.
Les noms de classes peuvent aussi être importés avec `clusterprep.models:ImageCNN`.

## Interfaces Python

```python
from clusterprep import (
    ExperimentConfig, TransferConfig, TransferPhase, run_experiment,
    resolve_transfer_splits, validate_transfer_modules,
    load_pretrained_modules_from_state,
)

config = ExperimentConfig(
    name="transfer", task="rr_rh", model="image_cnn",
    transfer=TransferConfig(
        source_experiment="results/source",
        module_prefixes={"radio": "encoder"},
        phases=[
            TransferPhase("head", 15, {"head": 1e-3}, patience=8),
            TransferPhase("all", 40, {"radio": 1e-5, "head": 1e-4}, patience=12),
        ],
    ),
)
# data_info = get_data_info_dict(...), avec les annotations actuelles
# run = run_experiment(data_info, config)
```

`TransferConfig` et `TransferPhase` vivent dans [transfer.py](src/clusterprep/transfer.py)
et sont exportés par `clusterprep`. Les dictionnaires imbriqués sont convertis
automatiquement par `ExperimentConfig`. Les noms de phases sont uniques et
ne peuvent pas contenir de chemin. Les phases et mappings vides sont refusés.

`Trainer.fit_transfer(train_loader, val_loader, *, run_dir=None, phases=None)`
réutilise `fit()` et `_epoch()`. Pour une utilisation directe, construire un
modèle cible, charger ses modules pré-entraînés, puis construire le `Trainer`.
`run_experiment()` réalise ces étapes automatiquement. `fit_transfer()` n'accepte
aucun DataLoader de test. Son retour est l'historique global ; les métadonnées
des phases sont disponibles dans `trainer.transfer_report`.

## Groupes et compatibilité

| Modèle | Groupes | Préfixe source radio habituel |
|---|---|---|
| SimpleCNN (`cnn`) | radio, head | encoder |
| ImageCNN | radio, head | encoder |
| FusionCNN | radio, xray, head | radio |
| DualSSN | radio, head | encoder |
| DualEncoderCNN | radio, head ; xray si présent | radio_encoder |
| DualEncoderSSN | radio, head ; xray si présent | radio_encoder |

Les couches de fusion apprenables existantes font partie du module `classifier`,
donc du groupe `head`. Cela préserve les clés des checkpoints et les `forward()`.
Un nouveau modèle peut déclarer un groupe `fusion` séparé. Ses groupes doivent
être des modules enregistrés et partitionner tous les paramètres sans recouvrement.
Les modèles à fusion précoce (SimpleCNN, ImageCNN, DualSSN) regroupent leur
encodeur complet sous `radio`, même lorsqu'il reçoit plusieurs modalités : ils
ne permettent pas de transférer séparément ses canaux radio/Chandra.

`module_prefixes` associe **groupe cible → préfixe du state_dict source**.
Les noms, formes et dtypes doivent tous correspondre, y compris les buffers
BatchNorm. Tous les mappings sont vérifiés avant de modifier un module cible.
La tête source est interdite. Les autres modules gardent leur initialisation.
L'API d'expérience vérifie également la tâche source, la taille d'image,
la normalisation, la structure des encodeurs, les canaux et le scattering.
Le transfert ImageCNN radio → branche radio DualEncoderCNN est possible si
les encodeurs correspondent ; les architectures complètes peuvent différer.

Formats acceptés : state_dict brut ou enveloppes `model_state_dict`,
`model_state`, `state_dict`, `model`, chargés avec `weights_only=True` sur CPU.
L'ancien ImageCNN (clés séparées `encoder.*`, `to_latent.*`, `FC_classifier.*`)
dispose d'une conversion explicite des deux branches de l'encodeur vers
`encoder.encoder.*` et `encoder.to_latent.*`. La correspondance exacte est
enregistrée ; aucun poids de la tête n'est importé.

Les métadonnées historiques de normalisation absentes sont signalées et les
valeurs historiques du projet sont supposées. Une incompatibilité déclarée
est refusée. L'empreinte SHA-256 du checkpoint source est sauvegardée.

## Phases, gel et sélection

Chaque phase crée AdamW avec ses propres groupes de learning rates. Les groupes
absents sont gelés et maintenus en mode `eval()` après chaque `model.train()` :
leurs statistiques BatchNorm et leur dropout restent fixes. Tous les groupes
nouvellement initialisés doivent être entraînés dès la première phase. Pour un
transfert radio seul vers un modèle multimodal, inclure `xray` et `head` dans
la première phase, ainsi que `fusion` si le modèle expose ce groupe.

`use_scheduler=true` crée un OneCycleLR neuf par phase. Chaque taux donné est
la référence de son groupe et `max_lr_factor` fixe son maximum ; désactiver le
scheduler pour des taux constants. Les paramètres globaux MixUp, clipping,
monitor et activation de l'early stopping s'appliquent aux phases. Les epochs,
patience et weight decay sont définis par phase. Le champ global `epochs` ne
limite pas la somme des phases.

Chaque phase restaure son meilleur checkpoint avant la suivante. La sélection
finale compare uniquement les meilleurs scores de validation, avec `val_loss`
par défaut et la première phase en cas d'égalité. `best.pt` peut donc provenir
d'une phase antérieure au fine-tuning. Les seuils RR/RH sont choisis sur la
validation finale ; le modèle sélectionné est évalué une seule fois sur test.
BCEWithLogitsLoss et sigmoid restent les opérations multi-label de référence.

## Splits et provenance

`resolve_transfer_splits()` est réutilisé sans régénérer de splits historiques.
Il retrouve les amas par `cluster_id`, conserve leurs affectations et ajoute
seulement les nouveaux cRR/cRH éligibles. Les annotations originales restent
dans `splits.json.samples[*].labels` et les targets viennent du catalogue actuel.
Cette résolution prend actuellement en charge `candidate_policy="positive"`.
La politique `ignore` reste disponible dans les entraînements classiques ou
avec `reuse_splits=false`, sans extension automatique.

Un catalogue avec plusieurs entrées pour le même `cluster_id` doit être
canonisé avant la résolution : les identités ambiguës sont refusées. Utiliser
`allow_subset=true` pour autoriser explicitement des exclusions historiques.
Les changements de chemins FITS sont signalés par le résolveur existant ; ils
ne constituent pas une preuve d'identité du contenu.

Des splits explicites peuvent être fournis s'ils correspondent aux splits
résolus. Ils ne les remplacent pas silencieusement. Avec `reuse_splits=false`,
mettre aussi `add_new_clusters=false` et `allow_subset=false` ; les splits
explicites ou générés restent contrôlés contre la source. Tout recouvrement du
train supervisé source avec la validation ou le test cibles est refusé. Un
recouvrement du test cible avec la validation source est signalé séparément,
car celle-ci a servi à la sélection du checkpoint source.

Les suites traitent chaque source séparément. La CLI produit des tableaux
séparés lorsque les populations de test diffèrent ; elle ne force pas une
comparaison entre des populations différentes.

## Artefacts et comparaison pilote

Chaque run sauvegarde `config.json`, `splits.json`, `split_distribution.json`,
`transfer.json`, `transfer_splits_report.json`, `best.pt`, les historiques CSV/JSONL,
les métriques et les prédictions validation/test. Chaque sous-dossier
`phases/<nom>/` contient `best.pt`, `history.csv`, `history.jsonl`, `metrics.json`.
L'historique global fournit `phase` et `phase_epoch` ; `plot_history()` affiche
les transitions. Les résultats test de transfert incluent les sous-populations
`historical`, `new_candidates` et `diffuse_emission` (DE explicite ou implicite).

Le [pilote](examples/transfer_pilot.py) compare trois runs indépendants sur les
mêmes splits : scratch, classifier-only, puis deux phases. Il utilise seulement
les API publiques existantes. Le bras scratch reçoit le budget cumulé des deux
phases et un LR de 1e-4. Il écrit les AP/F1 par label et par sous-population dans
`pilot_metrics.json`, ainsi que `comparison.csv` et les courbes par run.

```sh
# Pilote court, sans augmentation ; contrôle de fonctionnement, pas conclusion scientifique.
PYTHONPATH=clusterprep/src MPLBACKEND=Agg .venv/bin/python clusterprep/examples/transfer_pilot.py --head-epochs 2 --finetune-epochs 2 --no-augment

# Budget 15 + 40 epochs, augmentation et early stopping configurés dans le JSON.
PYTHONPATH=clusterprep/src MPLBACKEND=Agg .venv/bin/python clusterprep/examples/transfer_pilot.py

# Assistant interactif : choisir task=rr_rh, puis Training initialization=2.
PYTHONPATH=clusterprep/src .venv/bin/python -m clusterprep createconfig --output configs/transfer.json

# Tests synthétiques et non-régression, sans dépendre des vrais FITS.
PYTHONPATH=clusterprep/src MPLBACKEND=Agg .venv/bin/python -m pytest clusterprep/tests -q
```

Le pilote compare les trois modèles spécifiés à l'avance ; ses scores test ne
sont pas utilisés pour sélectionner un checkpoint ou régler les seuils. Des
répétitions avec plusieurs graines sont nécessaires pour interpréter une
différence de performances sur ce petit catalogue.
