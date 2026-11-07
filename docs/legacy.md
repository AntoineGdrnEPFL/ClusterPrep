# Intégration optique LEGACY

Les expériences RR/RH peuvent sélectionner `radio`, `xray` et `legacy` par JSON,
avec un CNN multicanal ou des encodeurs indépendants. Sans `modalities`, le
comportement historique des modes `raw`, `pair` et `multimodal` est conservé.

## Données inspectées

Inventaire local du 2 novembre 2026 : 273 FITS LEGACY, tous avec un HDU primaire
`(3, 512, 512)`, bandes `g/r/z` et WCS céleste TAN. Parmi les 236 amas classifiés,
206 possèdent LEGACY et 85 possèdent également CHANDRA. Les identifiants et labels
proviennent toujours de l'inventaire radio classifié.

La lecture accepte un HDU image 2D ou un cube `[bandes, y, x]`. Le premier HDU image
avec WCS céleste est choisi par défaut ; `legacy.hdu` permet une sélection explicite.
Les cubes de dimension supérieure sont refusés, sans sélection silencieuse du
premier plan. Les bandes sont sélectionnées par `BAND0`, `BAND1`, etc., ou par indices.

## Lancement depuis la racine PH_Project_I

```sh
.venv/bin/python -m clusterprep clusterprep/examples/legacy_multichannel.json
.venv/bin/python -m clusterprep clusterprep/examples/legacy_multimodal.json
.venv/bin/python -m clusterprep clusterprep/examples/legacy_comparison.json
```

Les chemins des JSON sont relatifs au répertoire de lancement. Adapter `device`
à `cpu` si nécessaire. Les exemples conservent les trois bandes optiques.

| Option dans chaque expérience | Valeur / effet |
| --- | --- |
| `modalities` | Liste ordonnée parmi `radio`, `xray`, `legacy` ; absente = historique |
| `missing_modality_policy` | `optional` (défaut), ou `required` |
| `include_masks` | `true` par défaut ; obligatoire pour `optional` |
| `legacy.raw_root` | Racine contenant les répertoires des amas |
| `legacy.filename_pattern` | Défaut `{cluster_id}_legacy.fits` |
| `legacy.bands` | Défaut `["g", "r", "z"]` ; noms ou indices, ordre conservé |
| `legacy.hdu` | Numéro ou nom du HDU ; absent = découverte |
| `legacy.normalization` | Paramètres optiques indépendants |
| `normalization.radio`, `normalization.xray` | Paramètres propres aux autres modalités |
| `model` | `cnn`, `image_cnn` ou `multi_encoder_cnn` avec modalités explicites |
| `model_params.fusion_hidden_dim` | Taille de la couche de fusion de `multi_encoder_cnn` |

Les options de normalisation sont `scope` (`image` ou `train`), `method`
(`percentile` ou `asinh_percentile`), `lower_percentile` (30), `upper_percentile`
(99), `alpha` (10). Les exemples utilisent `scope: "train"` pour LEGACY.
Les bornes sont calculées séparément par bande, uniquement sur les pixels finis
et couverts du sous-train, puis appliquées sans réestimation à validation/test.
Elles sont recalculées à chaque fold et au réentraînement final. Le mode `image`
est une transformation locale à chaque image, sans ajustement sur une population.

Chaque expérience conserve ses options, `normalization_bounds` et les clés du
train dans sa configuration/checkpoint et son rapport de normalisation. Pour une
inférence via `ClusterDataset`, passer les `normalization_bounds` de l'expérience ;
un dataset demandant des statistiques train sans bornes échoue explicitement.

## Canaux, disponibilité et modèles

Le tenseur `image` concatène, dans l'ordre de `modalities`, les bandes d'une
modalité puis ses masques. Radio conserve un canal ; X-ray a un canal et un masque ;
LEGACY conserve un masque par bande, car leur couverture peut différer.
Ainsi, trois bandes optiques donnent **9 canaux** au total avec les masques,
et **5 canaux** avec `legacy.bands: ["r"]`.

Pour exactement trois canaux RAW + CHANDRA + LEGACY : sélectionner `["r"]`,
`include_masks: false` et `missing_modality_policy: "required"`. Pour deux canaux
RAW + LEGACY, retirer `xray` de cette même configuration.

Chaque sample conserve `raw`, `chandra`, `legacy` et leurs masques pour les
modalités sélectionnées, `cluster_id`, `key`, les cibles historiques, ainsi que
`has_chandra`, `has_legacy` et `availability` dans l'ordre de `modalities`.
Une observation absente produit des zéros avec masque faux ; un zéro physique
valide garde un masque vrai. Les rotations, flips et zooms utilisent les mêmes
indices et la même grille pour toutes les bandes et masques. L'augmentation
explicite est réservée à `training=True`.

`image_cnn` et `cnn` déduisent le nombre de canaux du choix des bandes.
`multi_encoder_cnn` réutilise `ImageEncoder` dans un `ModuleDict`, un encodeur par
modalité, et concatène les latents avec les fractions de couverture avant la tête.
Un latent sans couverture est annulé. Les anciens modèles et leurs noms de
paramètres restent disponibles sans `modalities`.

## Prétraitement et cache

LEGACY est reprojeté avec le mécanisme `reproject_interp` existant sur la fenêtre
radio **avant** la moyenne en blocs `2^n`. Le crop, le facteur et le WCS final sont
identiques à ceux de RAW/CHANDRA. Les NaN et pixels hors couverture restent invalides
jusqu'à la normalisation. Un bloc est couvert s'il contient au moins un pixel fini.

Les FITS physiques LEGACY et LEGACY_MASK sont enregistrés séparément, avec WCS,
facteur, provenance et checksums, selon les conventions du cache existant.
Le choix HDU/bandes fait partie du nom et de la provenance ; taille, version et
signatures des sources radio/optique sont vérifiées à la lecture. Les options de
normalisation ne nécessitent pas de recalcul du cache physique.

Pour préparer ou régénérer seulement LEGACY :

```sh
.venv/bin/python -m clusterprep preprocess scratch/data/PSZ2/classified \
  --legacy-root scratch/data/PSZ2/fits --legacy-only \
  --legacy-bands g r z --size 128 --processed-dir processed
```

Ajouter `--force` pour régénérer l'optique. Cette commande n'écrit aucun produit
RAW/CHANDRA. L'API équivalente est `load_processed_optical(..., force=True)`.

## Splits et comparaisons

La sélection des modalités ne change pas la stratification, le seed ni les
partitions de base. `required` filtre chaque partition après affectation et écrit
les exclusions dans `modality_exclusions.json`. En CV, les folds de base restent
identiques ; les effectifs réellement évalués pondèrent l'agrégation et seuls les
amas conservés sont attendus dans les prédictions hors fold. Un sous-train devenu
invalide provoque une erreur, sans réaffectation automatique.

La suite [legacy_comparison.json](../examples/legacy_comparison.json) lance les
quatre combinaisons sur les mêmes partitions, avec la politique `optional`.
Pour réutiliser des partitions enregistrées, fournir les trois listes de clés
train/validation/test via `split.splits` dans le JSON de suite. Pour une comparaison
sur un sous-ensemble `required`, fournir ces mêmes listes filtrées à **toutes** les
références. Les outils de comparaison continuent de refuser des cohortes différentes.
Les CSV de comparaison ajoutent la colonne `modalities` aux métriques habituelles.
Aucun fichier de split ou checkpoint existant n'est modifié.

## Transfer learning

Les groupes logiques sont `radio`, `xray`, `legacy`, `head`. Les aliases
`optical_encoder` et `legacy_encoder` pointent vers `legacy`. Le transfert modulaire
existant accepte par exemple, dans l'expérience `multi_encoder_cnn` :

```json
{
  "transfer": {
    "enabled": true,
    "sources": {"old": {"checkpoint": "results/source/best.pt"}},
    "defaults": {"on_incompatible": "error", "on_missing": "error"},
    "mappings": [
      {"source": "old", "from": "radio", "to": "radio"},
      {"source": "old", "from": "xray", "to": "xray"}
    ]
  },
  "training_phases": [
    {"name": "optical_and_head", "epochs": 5,
     "learning_rates": {"legacy": 0.001, "head": 0.001}},
    {"name": "finetune", "epochs": 20,
     "learning_rates": {"radio": 0.00001, "xray": 0.00001,
                        "legacy": 0.0001, "head": 0.0001}}
  ]
}
```

La source doit posséder des encodeurs compatibles (par exemple
`dual_encoder_cnn`, même taille et branche X-ray avec masque). L'optique est
initialisée indépendamment. Aucune copie radio→optique n'est implicite.

Pour un CNN multicanal dont la première convolution change, la politique explicite
est de **conserver son initialisation neuve** : mapper cette couche avec
`on_incompatible: "skip"` et mapper séparément les couches compatibles, ou omettre
le mapping de l'encodeur entier. Pour `image_cnn`, la première convolution est
`radio.encoder.0` ; `radio.to_latent` peut être transféré séparément. Le rapport de
transfert consigne les mappings ignorés. Aucun élargissement implicite des poids.

## Visualisation

`plot_sample(sample, chandra_contours=True)` affiche radio, X-ray, LEGACY puis les
masques. LEGACY est affiché comme moyenne des bandes normalisées ; les bandes
restent séparées pour le modèle. L'orientation est commune et les observations
absentes sont annotées. Les contours CHANDRA sont optionnels.

## Validation et limites

Les tests ciblés sont dans [test_legacy.py](../tests/test_legacy.py).
Le script [legacy_smoke.py](../examples/legacy_smoke.py) réalise un test réel :

```sh
.venv/bin/python clusterprep/examples/legacy_smoke.py \
  scratch/data/PSZ2/classified scratch/data/PSZ2/fits
```

Il utilise un nouveau dossier temporaire pour le cache, les checkpoints et une
figure. Essai réalisé sur PSZ2G023.17+86.71, PSZ2G031.93+78.71 et PSZ2G040.58+77.12 :
batch `3×9×128×128`, forward/backward et étape AdamW réussis pour `image_cnn` et
`multi_encoder_cnn`, logits identiques après rechargement, trois caches réutilisés.
Ce test ne mesure pas la performance scientifique d'une expérience complète.

Résultat de la suite complète (`python -m pytest clusterprep/tests -q`) :
**468 réussis, 1 ignoré, 3 échecs, 33 sous-tests réussis**. Les 14 nouveaux tests
LEGACY passent. Les trois échecs ont été reproduits séparément sur une extraction
de la version initiale `76a8e7c` : deux tests cherchent les configurations binaires
à un ancien emplacement (`configs/rr_binary.json` et `configs/rh_binary.json`), et
le test du notebook cherche un FITS prétraité synthétique absent. Ils ne sont pas
corrigés par cette intégration. `git diff --check` ne signale aucune erreur.

Limites explicites : la grille radio reste nécessaire même pour une expérience
optique seule ; les radios plus petites que la taille demandée restent refusées
comme dans la pipeline historique. Pas de nouveau support LEGACY pour scattering
précalculé ou fusions spatiales spécifiques à deux modalités. Les bornes globales
nécessitent au moins un pixel couvert par bande dans le train. Les bandes réparties
dans plusieurs HDU doivent être préparées en cube ou sélectionnées par HDU ; elles
ne sont pas fusionnées implicitement. La normalisation train rassemble en mémoire
les pixels valides des images réduites. Les notebooks spécialisés de galerie
bimodale ne sont pas réécrits ; utiliser `plot_sample` pour les trois modalités.

## Fichiers de l'intégration

- [io.py](../src/clusterprep/io.py), [preprocessing.py](../src/clusterprep/preprocessing.py), [processed.py](../src/clusterprep/processed.py) : découverte, lecture, grille et cache.
- [modalities.py](../src/clusterprep/modalities.py), [dataset.py](../src/clusterprep/dataset.py) : sélection, bandes, normalisation et augmentation.
- [models.py](../src/clusterprep/models.py), [modular_transfer.py](../src/clusterprep/modular_transfer.py) : encodeurs variables et mappings.
- [training.py](../src/clusterprep/training.py), [cross_validation.py](../src/clusterprep/cross_validation.py) : configuration, train-only, exclusions et comparaison.
- [visualization.py](../src/clusterprep/visualization.py), [__main__.py](../src/clusterprep/__main__.py), [__init__.py](../src/clusterprep/__init__.py) : affichage, CLI et API.
- [legacy_multichannel.json](../examples/legacy_multichannel.json), [legacy_multimodal.json](../examples/legacy_multimodal.json), [legacy_comparison.json](../examples/legacy_comparison.json), [legacy_smoke.py](../examples/legacy_smoke.py) : exemples.
- [test_legacy.py](../tests/test_legacy.py), cette documentation et [README.md](../README.md).
