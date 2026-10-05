# Entraîner et comparer plusieurs expériences

Pour construire un JSON sans l'écrire à la main :

```sh
python -m clusterprep createconfig
# Ou choisir le fichier de sortie dès le lancement :
python -m clusterprep createconfig --output configs/mes_experiences.json
```

L'assistant demande les chemins, classes, splits (ou fold de test), puis les
paramètres de chaque expérience : mode, modèle, graine, taille des images/lots,
nombre d'époques, learning rate, régularisation, augmentation, caches, workers,
device et early stopping. `model_params` est saisi comme un objet JSON.
Entrée accepte la valeur proposée ; les défauts suivent `ExperimentConfig`.
Il est possible d'ajouter plusieurs expériences dans le même fichier.

Les saisies et constructeurs de modèles sont validés, sans charger les FITS ni
lancer d'entraînement. Le device choisi est enregistré pour la machine cible ;
sa disponibilité matérielle n'est pas vérifiée. Les chemins sont enregistrés
en absolu relativement au dossier courant. Un fichier existant n'est remplacé
qu'après confirmation ; Ctrl+C ou fin d'entrée annule la création.
La cohorte proposée est `paired` si une expérience utilise le mode paire,
sinon `all`. Les effectifs suffisants pour les splits seront vérifiés au training.

L'API Python équivalente est `clusterprep.config_wizard.create_config()` ; elle
retourne le chemin créé, ou `None` en cas d'annulation.

Depuis `PH_Project_I`, installer avec `python -m pip install -e ./clusterprep`,
puis lancer la suite d'exemple :

```sh
python -m clusterprep clusterprep/examples/experiments.json
```

Les chemins du JSON sont relatifs au répertoire de lancement. Les expériences
sont exécutées séquentiellement dans des dossiers uniques sous
`results/experiments`. Le JSON fourni lance jusqu'à 50 époques par expérience.

## Cache FITS et progression (v0.3)

Les données traitées sont conservées dans `processed/<cluster_id>/`, partagé
entre expériences. Ce dossier est relatif au répertoire de lancement.
Pour choisir un autre emplacement :

```sh
python -m clusterprep clusterprep/examples/experiments.json --processed-dir processed
```

Ou utiliser `ExperimentConfig(..., processed_dir="processed", verbose=True)`.
Dans le JSON, ces deux champs se placent dans chaque expérience.
`processed_dir=null` désactive le cache disque. Le paramètre `cache` contrôle
séparément le cache mémoire des tenseurs normalisés.

Les images sont enregistrées séparément dans le HDU primaire de trois FITS :

```text
processed/<cluster_id>/<cluster_id>_RAW_processed_size128_v2.fits
processed/<cluster_id>/<cluster_id>_CHANDRA_processed_size128_v2.fits
processed/<cluster_id>/<cluster_id>_CHANDRA_MASK_processed_size128_v2.fits
```

La taille et la version du preprocessing permettent de conserver plusieurs
configurations côte à côte. Le mode radio utilise seulement RAW ; les modes
pair et multimodal partagent ce même fichier RAW. Le masque est une image
uint8 (0/1). Chaque fichier porte le WCS de la grille radio réduite.
**Aucune normalisation ni augmentation n'est enregistrée** : les valeurs et NaN
sont conservés. Le Dataset normalise puis augmente après lecture des FITS.

Les métadonnées de provenance incluent chemins sources, tailles, dates de
modification, taille cible et version du prétraitement. Les sources doivent
rester accessibles pour vérifier cette provenance. Les fichiers valides sont
conservés ; les absents ou invalides sont préparés automatiquement. Si les
sources changent, les fichiers concernés sont remplacés. Chaque FITS est écrit
atomiquement avec checksum ; un fichier corrompu est reconstruit. Les anciens
caches aux noms `raw_<size>_<signature>.fits` / `pair_<size>_<signature>.fits`
ne sont plus lus ; ils restent sur disque et le nouveau format est recalculé.

### Prétraitement indépendant

Aucun JSON d'expériences ni entraînement n'est nécessaire :

```sh
python -m clusterprep preprocess /chemin/vers/data --processed-dir processed --size 128
# Recalcul explicite, même si les fichiers sont valides :
python -m clusterprep preprocess /chemin/vers/data --processed-dir processed --size 128 --force
```

Le dossier source contient `RAW/<classe>/` et `CHANDRA/<classe>/` comme pour
l'entraînement. `--mode multimodal` (défaut) prépare RAW pour tous les amas et
CHANDRA + CHANDRA_MASK lorsqu'une observation Chandra existe. `--mode raw`
prépare uniquement la radio ; `--mode pair` sélectionne uniquement les amas
avec Chandra. `--class-names DE NDE` configure les classes, `--quiet` masque
la progression. Un bilan JSON est affiché à la fin.

La préparation précède la création des workers d'entraînement. Les compteurs
de créations, réutilisations et reconstructions ainsi que la durée sont
enregistrés dans `preprocessing.json` et `metrics.json` de chaque run.
Pour préparer ou lire le cache directement :

```python
from clusterprep import prepare_processed, load_processed
prepare_processed(info, list(info), mode="multimodal", size=128,
                  processed_dir="processed", force=False)
images = load_processed(info[next(iter(info))], mode="raw", size=128)
# images["raw"] contient les valeurs traitées, non normalisées.
```

En terminal interactif, une barre transitoire affiche le prétraitement, puis
les lots train/validation/test. Une ligne par époque résume pertes, F1,
checkpoint/patience et durée. Dans les logs redirigés, seuls ces résumés sont
émis, sans barres ni retours chariot. Les avertissements répétitifs
`FITSFixedWarning` sont regroupés dans `fits_warnings.log` du run ; les autres
avertissements restent visibles. `verbose=False` ou `--quiet` masque barres et
résumés. La CLI conserve son JSON final indiquant les chemins des résultats.

## Trois modes de données

| Mode | Amas utilisés | Entrée du modèle |
| --- | --- | --- |
| `raw` | Tous les amas radio | Radio, 1 canal |
| `pair` | Amas avec radio et fichier Chandra | Radio + Chandra + masque, 3 canaux |
| `multimodal` | Tous les amas radio, appariés ou non | Même entrée à 3 canaux ; Chandra et masque nuls si absents |

`pair` est déjà multimodal au sens des entrées. `multimodal` étend ce mode aux
observations incomplètes pour entraîner avec les deux populations. Un fichier
Chandra présent mais sans recouvrement produit également un masque nul.
La cohorte `paired` signifie « fichier Chandra disponible », pas « couverture
non nulle ». `has_chandra` et le masque permettent de distinguer ces cas.

Deux modèles de référence sont fournis : `cnn`, à concaténation des canaux,
et `fusion`, avec deux branches radio/X. `fusion` utilise aussi la fraction de
couverture et annule les caractéristiques X quand le masque est entièrement
nul. Il accepte les modes `pair` et `multimodal`. Ce sont de petits modèles
pour servir de point de départ, sans poids préentraînés téléchargés.

## API Python et plusieurs graines

```python
from dataclasses import replace
from clusterprep import (
    ExperimentConfig, get_data_info_dict, run_experiments, compare_experiments,
)

info = get_data_info_dict("scratch/data/PSZ2/classified")
base = ExperimentConfig(
    name="radio", mode="raw", model="cnn", size=128,
    epochs=50, batch_size=8, learning_rate=1e-3,
    patience=8, min_delta=1e-4, monitor="val_loss",
    class_weight="balanced", device="cpu",
)
variants = [
    base,
    replace(base, name="pairs", mode="pair"),
    replace(base, name="multimodal", mode="multimodal", model="fusion"),
]
configs = [replace(c, name=f"{c.name}_seed{seed}", seed=seed)
           for c in variants for seed in (42, 123, 456)]
runs = run_experiments(info, configs, split_seed=42,
                       output_dir="results/my_experiments")
rows = compare_experiments(runs, "results/my_experiments/comparison.csv",
                           cohort="paired")
```

La graine de séparation (`split_seed`) est indépendante de la graine du modèle
(`seed`). Une suite utilise les mêmes splits radio pour tous les modèles, puis
filtre chaque split pour `pair`. Aucun amas ne passe d'un split à un autre.
Une classe manquante dans l'entraînement après filtrage provoque une erreur.
Les augmentations ne concernent que l'entraînement. Les graines initialisent
Python, NumPy, PyTorch, les workers et le mélange des lots. Les opérations GPU
ne sont pas forcées en mode déterministe : l'identité bit à bit est testée
sur CPU, pas garantie entre machines ou accélérateurs.

`run_experiment` accepte une configuration unique et des `splits=(train, val,
test)` explicites. Sans splits, il utilise la séparation radio avec la graine
42. `run_experiments` accepte aussi `n_folds` et `fold_index` : exécuter un appel
par fold. Les comparaisons directes entre folds différents sont refusées,
car ils n'ont pas les mêmes amas de test ; leurs scores restent enregistrés
pour une analyse de cross-validation séparée.

## Early stopping et checkpoint

Le suivi porte uniquement sur la validation : `val_loss` à minimiser, ou
`val_macro_f1` / `val_balanced_accuracy` à maximiser. La première époque est
enregistrée. Une amélioration doit dépasser strictement `min_delta` ; après
`patience` époques consécutives sans cette amélioration, l'entraînement s'arrête.
`best.pt` correspond au dernier meilleur score accepté selon cette règle.
Une copie profonde du meilleur état est restaurée en mémoire avant l’évaluation
finale, jamais sélectionnée selon le test ; `best.pt` reste disponible sur disque.
Le test est évalué une seule fois, puis ses sous-cohortes sont calculées à partir
des mêmes prédictions. Pour ne pas arrêter avant la limite, utiliser une
`patience` au moins égale à `epochs`.

`TrainingConfig` centralise le protocole, hérité par `ExperimentConfig`.
`LEGACY_TRAINING_CONFIG = TrainingConfig()` fournit les valeurs de référence :
200 époques, batch 16, AdamW (`learning_rate=5e-5`, `weight_decay=0.1`),
CrossEntropyLoss avec poids `N_total / N_classe` calculés sur le train seul et
`label_smoothing=0.1`, OneCycleLR (`max_lr_factor=3`, `warmup_fraction=0.3`,
cosinus), MixUp (`mixup_probability=0.5`, `mixup_alpha=0.4`), clipping à 1,
early stopping sur `val_loss` avec patience 50 et `min_delta=0`.
Le nom `learning_rate` est conservé dans l’API et les JSON existants.
**Les configurations anciennes qui omettent ces champs prennent désormais ces
nouveaux défauts** ; les résultats précédents ne décrivent donc pas ce protocole.

La même CrossEntropyLoss est utilisée au train, en validation et au test.
La loss par époque vaut `sum(batch_loss * batch_size) / sum(batch_size)`.
Avec les poids, `batch_loss` suit exactement la réduction moyenne de PyTorch
(diviseur égal à la somme des poids des cibles). Cette agrégation demandée
n’est donc pas une CE pondérée recalculée en un seul batch sur tout le dataset.
La colonne `loss` des CSV de prédictions et les losses des cohortes restent
la CE individuelle **sans pondération ni smoothing**, pour permettre les
comparaisons. Elles diffèrent de la loss surveillée ; le schéma des nouveaux
runs est versionné 2.

Le scheduler avance après `optimizer.step()` à chaque batch. MixUp utilise
la même permutation et le même lambda pour toutes les modalités, masque
compris (il devient souple). Les métriques train comparent les prédictions
sur entrées éventuellement mélangées aux labels originaux : elles sont
indicatives. Validation et test utilisent `eval()` et `inference_mode()`,
sans augmentation ni MixUp.

Le train mélange les exemples et utilise `drop_last_train=True`. Validation
et test gardent tous les exemples dans l’ordre. Avec `drop_last_train=False`,
un éventuel dernier singleton est fusionné avec le lot précédent si le modèle
utilise BatchNorm. Si aucun batch train ne reste, une erreur explicite invite
à diminuer la taille du batch ou à désactiver `drop_last_train`.

Pour les ablations, modifier séparément `use_mixup=False`,
`use_class_weights=False`, `label_smoothing=0`, `use_scheduler=False`,
`gradient_clip_norm=0`, ou `weight_decay=0`.
L’ancien champ `class_weight="none"` désactive également les poids ;
`"balanced"` laisse `use_class_weights` décider.
`use_early_stopping=False` entraîne toutes les époques, mais restaure toujours
le meilleur modèle de validation. Les autres moniteurs existants restent
accessibles pour des expériences hors preset legacy.

### Baseline radio ImageCNN

```sh
python -m clusterprep clusterprep/examples/legacy_image_cnn_radio.json
```

Ce fichier explicite tous les paramètres du protocole et utilise le modèle
`image_cnn` radio, sur MPS (remplacer par `cpu` ou `cuda` selon la machine).
La reproduction numérique des scores legacy nécessite encore une comparaison
sur les mêmes données, splits, preprocessing, architecture et graines.
Les tests de la bibliothèque vérifient le protocole, pas cette équivalence
scientifique.

### Trainer et adaptateurs

`Trainer` expose `train_one_epoch`, `evaluate`, `fit` et `predict`.
`run_experiment` / `run_experiments` acceptent `input_adapter(batch)` en plus
de `model_factory`. L’adaptateur sélectionne exclusivement les entrées du
modèle : un tenseur, un tuple pour `model(*inputs)` ou un dictionnaire pour
`model(**inputs)`. Cela permet notamment un modèle X-ray seul ou deux encodeurs :

```python
run_experiment(info, config, model_factory=build_two_encoders,
               input_adapter=lambda b: (b["raw"], b["chandra"], b["chandra_mask"]))
```

Pour un modèle X-ray seul, choisir le mode `pair` et un adaptateur retournant
`b["chandra"]`, avec une factory construisant un modèle à un canal.
Chaque appel à la factory doit créer un modèle neuf ; les graines sont fixées
avant son appel. Le Trainer peut aussi recevoir des DataLoaders externes dont
les batches contiennent `label` et les modalités choisies par l’adaptateur.
Les identifiants d’amas ne sont alors pas obligatoires ; `predict` n’exige
pas de labels. Créer un modèle et un Trainer neufs pour chaque seed/fold.

## Fichiers conservés pour chaque expérience

| Fichier | Contenu |
| --- | --- |
| `config.json` | Hyperparamètres, classes, modèle, effectifs, versions des dépendances, empreinte du code |
| `splits.json` | Splits de base et effectifs utilisés, informations des amas, chemins absolus et signatures des FITS |
| `history.csv` | Métriques scalaires train/validation par époque, temps et learning rate |
| `history.jsonl` | Même historique avec matrices de confusion et métriques par classe |
| `best.pt` | Poids du modèle, état optimiseur, configuration, classes, époque et score sélectionnés |
| `metrics.json` | Scores finaux, sous-cohortes, durées, époque sélectionnée et arrêt anticipé |
| `validation_predictions.csv`, `test_predictions.csv` | Clé, amas, classe vraie/prédite, présence Chandra, perte et probabilités par classe |
| `status.json` | État running/completed/failed, avec erreur si échec |

Les métriques comprennent accuracy, balanced accuracy, F1 macro et pondéré,
precision/recall/F1/support par classe, ROC-AUC et average precision un-contre-
tous, ainsi que la matrice de confusion. L'ordre des classes est enregistré
(DE=0, NDE=1 par défaut). La matrice a les classes vraies en lignes et prédites
en colonnes. Une AUC/AP indéfinie est `null` en JSON et vide en CSV ; les
divisions par zéro de precision/recall/F1 donnent zéro. Le F1 macro inclut toutes
les classes configurées, même absentes d'un sous-ensemble ; la balanced accuracy
moyenne les rappels des classes présentes.

`compare_experiments` compare la cohorte `paired` par défaut, ou `all` /
`raw_only` explicitement. Il refuse une cohorte vide ou différente entre runs.
L'identité vérifiée inclut les clés, amas, labels, ordre des classes et signatures
des fichiers (chemin, taille et date de modification ; pas de hash du contenu).
Les populations d'entraînement peuvent différer : la table enregistre le mode
et les manifests gardent les effectifs. Pour une ablation sur exactement les
mêmes populations, fournir des splits contenant seulement les amas appariés
aux trois modes.

## Modèle personnalisé et rechargement

Le [guide des modèles](MODELS.md) décrit le registre `register_model`, les
paramètres `model_params` et l'import direct `module:Constructeur` depuis JSON.
`ImageCNN` est intégré sous le nom `image_cnn` ; une suite prête à lancer est
disponible dans `examples/image_cnn.json`.

Passer `model_factory(config, num_classes)` à `run_experiment` ou
`run_experiments`. La factory est appelée après initialisation de la graine et
doit créer un modèle neuf prenant un tenseur `[B, C, H, W]` et retournant des
logits `[B, num_classes]`. C vaut 1 pour `raw`, 3 sinon. Nommer `config.model`
pour identifier l'architecture personnalisée. L’optimiseur fourni est AdamW.

```python
import torch
from clusterprep import ExperimentConfig, build_model

checkpoint = torch.load("chemin/vers/best.pt", map_location="cpu", weights_only=True)
config = ExperimentConfig(**checkpoint["config"])
model = build_model(config, len(checkpoint["class_names"]))  # ou votre factory
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

Pour tracer les courbes : `fig = plot_history(run_dir, "curves.png")` depuis
`clusterprep`, puis fermer la figure avec `matplotlib.pyplot.close(fig)`.

Le cache mémoire conserve les tenseurs normalisés avant augmentation, par Dataset et
par worker. Pour limiter la mémoire, garder `num_workers=0` ou désactiver
`cache`. Le prétraitement FITS précède les époques et son résultat est partagé
entre entraînements via le cache disque. La première époque charge et normalise
les petits fichiers traités pour remplir le cache mémoire.
Le dispositif par défaut est `mps` ; `cpu` ou `cuda` peuvent être spécifiés si
disponibles dans votre environnement.

## Validation rapide

```sh
PYTHONPATH=clusterprep/src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/clusterprep-mpl .venv/bin/python -m unittest discover -s clusterprep/tests -v
PYTHONPATH=clusterprep/src MPLCONFIGDIR=/tmp/clusterprep-mpl .venv/bin/python clusterprep/examples/check_training.py scratch/data/PSZ2/classified
```

Le second script sélectionne 20 amas réels, avec et sans Chandra, répartis en
12/4/4. Il exerce les trois modes à 32 × 32 pixels et force l'arrêt après deux
époques (`min_delta=100`, exclusivement pour ce test). Il produit les checkpoints,
les métriques, une comparaison CSV et les courbes d'apprentissage. Ces scores
vérifient le fonctionnement de la chaîne ; ils ne mesurent pas les performances
d'un modèle entraîné pour l'étude scientifique.
