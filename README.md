# clusterprep

Bibliothèque Python autonome extraite de `dcreclass/scripts/03.create_processed_data.ipynb`.
Elle n'importe pas `dcreclass` et ne modifie ni les FITS ni le notebook source.
Le code adapté du notebook conserve la licence MIT fournie dans `LICENSE`.

La version 0.2 ajoute l'entraînement multi-expériences : voir
[le guide d'entraînement](TRAINING.md) pour les trois modes de données,
l'early stopping, les checkpoints et la comparaison des métriques.

## Installation

Depuis la racine `PH_Project_I`, dans votre environnement Python :

```sh
python -m pip install -e ./clusterprep
```

Les dépendances sont déclarées dans `pyproject.toml`, dont `reproject` et PyTorch.
Pour utiliser l'environnement déjà présent sans installation :

```sh
PYTHONPATH=clusterprep/src dcreclass/.venv/bin/python votre_script.py
```

## Entraînement

```python
import torch
from torch.utils.data import DataLoader
from clusterprep import CLASS_NAMES, get_data_info_dict, split_data_keys, ClusterDataset

info = get_data_info_dict("scratch/data/PSZ2/classified")
train_keys, val_keys, test_keys = split_data_keys(info, mode="pair", seed=42)
train = ClusterDataset(info, train_keys, CLASS_NAMES, mode="pair", size=128, augment=True)
val = ClusterDataset(info, val_keys, CLASS_NAMES, mode="pair", size=128)
test = ClusterDataset(info, test_keys, CLASS_NAMES, mode="pair", size=128)
loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=0)

batch = next(iter(loader))
# Entrée à trois canaux pour un modèle choisi par l'utilisateur :
x = torch.cat([batch["raw"], batch["chandra"], batch["chandra_mask"]], dim=1)
y = batch["label"]  # int64, DE=0, NDE=1
# logits = model(x)
# loss = torch.nn.functional.cross_entropy(logits, y)
# loss.backward()
```

`mode="raw"` accepte tous les FITS radio ; `mode="pair"` sélectionne ceux ayant
un fichier Chandra. Structure attendue : `RAW/{DE,NDE}/*.fits` et
`CHANDRA/{DE,NDE}/<nom_radio>CHANDRA.fits`. Les doublons de noms entre classes
sont refusés pour éviter un écrasement silencieux de l'inventaire.

Chaque échantillon contient `raw` de forme `(1, size, size)`, `label` et
`cluster_id`. Le mode paire ajoute `chandra` et `chandra_mask` de même forme.
Les images sont en float32, normalisées dans `[0, 1]` par percentiles 30–99
puis asinh. Les pixels invalides sont remplis par zéro ; le masque distingue
la couverture Chandra. Un bloc est valide s'il contient au moins un pixel fini.
Une paire sans recouvrement est conservée avec un masque et une image Chandra nuls.

Le traitement sélectionne un crop radio centré, réduit par une puissance de
deux, après reprojection de Chandra sur cette même fenêtre. Une image radio
trop petite est refusée. Le premier HDU contenant une image avec WCS céleste
est utilisé ; pour un cube non singleton, le premier plan est sélectionné
avec avertissement. Il ne s'agit pas d'une intégration de cube spectral.

Les splits sont stratifiés et regroupés par `cluster_id`, avant augmentation.
Les petites classes peuvent empêcher une stratification ; l'erreur est explicite.
`n_folds=10, fold_index=0` sélectionne un fold de test et remplace `test_size`.
`val_size` reste la fraction du jeu complet (par défaut 20 %).
Chaque amas d'entraînement possède 24 variantes : 12 rotations par pas de
30 degrés, avec ou sans retournement horizontal, appliquées à tous les canaux.
Ne pas activer `augment` pour la validation ou le test.

Le preprocessing peut être lancé indépendamment de l'entraînement :

```sh
python -m clusterprep preprocess /chemin/vers/data --processed-dir processed --size 128
```

Par défaut, toutes les radios et les observations Chandra disponibles sont
préparées. `--mode raw` limite le travail aux radios ; `--mode pair` sélectionne
les amas avec Chandra. Les résultats sont enregistrés avant normalisation et
augmentation dans `processed/<cluster_id>/` :

- `<cluster_id>_RAW_processed_size128_v2.fits`
- `<cluster_id>_CHANDRA_processed_size128_v2.fits`
- `<cluster_id>_CHANDRA_MASK_processed_size128_v2.fits`

Chaque FITS contient une image dans son HDU primaire et le WCS adapté ; le
masque contient des 0/1. Le nom distingue la taille cible et la version pour
conserver plusieurs configurations. L'entraînement réutilise ces fichiers et
prépare automatiquement les absents ou invalides. `--force` force le recalcul
du preprocessing ; l'API expose aussi `prepare_processed(..., force=True)` et
`load_processed(..., force=True)`. `processed_paths(...)` fournit les chemins
des images ; `processed_path(...)` fournit celui de RAW, y compris en mode pair.

`processed_dir` configure le dossier partagé (`None` pour désactiver le cache
disque). Des barres de progression et un résumé par époque suivent le travail ;
`--quiet` masque cet affichage. Voir [TRAINING.md](TRAINING.md) pour la provenance
et le passage depuis l'ancien cache.

Le chargement est paresseux. `ClusterDataset(..., cache=True)` conserve les
images normalisées avant augmentation ; l'entraînement active ce cache par
défaut. Sans cache mémoire, chaque accès relit les petits FITS traités.
La reprojection est recalculée uniquement si le cache disque manque ou est invalide.
La mémoire temporaire dépend de la taille de la fenêtre radio avant réduction.

## Visualisation

Le notebook [analyse_cluster_radio_chandra.ipynb](notebooks/analyse_cluster_radio_chandra.ipynb)
permet de choisir `CLUSTER_ID`, puis d'examiner les FITS natifs, le crop radio,
la reprojection Chandra, la réduction, la normalisation et la couverture.
Il propose des histogrammes, des statistiques et un export facultatif des figures.
Ouvrir le notebook avec le même environnement Python que la bibliothèque et
exécuter toutes les cellules après modification des paramètres. Il prend aussi
en charge les amas sans Chandra et les images sans recouvrement.

```python
import matplotlib.pyplot as plt
from clusterprep import plot_sample

fig = plot_sample(train[0], "results/sample.png")
plt.close(fig)
```

`plot_comparison` compare aussi les deux FITS natifs et Chandra reprojeté avec
leurs axes célestes. Les fonctions retournent la figure ; l'appelant la ferme.
La bibliothèque ne force pas le backend Matplotlib.

## Tests reproductibles

Depuis `PH_Project_I` :

```sh
PYTHONPATH=clusterprep/src MPLCONFIGDIR=/tmp/clusterprep-mpl dcreclass/.venv/bin/python -m unittest discover -s clusterprep/tests -v
PYTHONPATH=clusterprep/src MPLCONFIGDIR=/tmp/clusterprep-mpl dcreclass/.venv/bin/python clusterprep/examples/check_data.py scratch/data/PSZ2/classified --limit 0
```

Les tests synthétiques vérifient les extensions FITS, l'alignement, l'absence
de recouvrement, les NaN, les lots, les augmentations synchronisées, les figures
et l'absence de fuite entre splits. Le script réel charge tous les échantillons
avec `--limit 0` (4 par mode par défaut), vérifie formes et valeurs finies,
et écrit six PNG ainsi que les mesures dans `results/clusterprep/report.json`.
Les temps mesurés excluent la génération des figures.

Validation locale du 28 septembre 2026 : les cinq tests passent et le paquet
wheel a été construit et installé dans un répertoire temporaire. Le chargement
complet a réussi pour 207 images radio (67 DE, 140 NDE) en 9,4 s et 85 paires
radio/Chandra en 63,0 s, à 128 × 128 pixels. Splits train/validation/test :
123/42/42 en radio et 51/17/17 en paire. Six figures ont été produites ; une
figure radio et une figure paire ont été inspectées visuellement. Astropy émet
des avertissements de correction des métadonnées FITS (dates et unités),
sans erreur de chargement. Ces mesures ne constituent pas un benchmark mémoire.

## Analyser et comparer les entraînements

Ouvrir [comparer_entrainements.ipynb](notebooks/comparer_entrainements.ipynb)
avec le kernel Python du projet. Il découvre les runs sous `results/clusterprep`
(parcours récursif, racines configurables), propose une sélection multiple et
fournit les diagnostics détaillés de chaque entraînement, les classements sur
la validation, une synthèse entre splits et l'évaluation sur le test.

La comparaison Chandra associe uniquement les mêmes familles de modèles,
contrôle les hyperparamètres/splits/sources et recalcule les deltas sur les mêmes
amas. Les intervalles bootstrap sont appariés par amas. Les modèles sans
référence radio seule sont explicitement signalés. Les exports CSV et le
manifeste de sélection sont activables dans la première cellule.

Conserver `notebooks/training_analysis.py` à côté du notebook. L'analyse ne
charge ni FITS ni checkpoints et ne requiert pas de GPU. Dépendances : `numpy`,
`pandas`, `matplotlib`, `scikit-learn`, `ipython` ; `ipywidgets` est facultatif
(une sélection par noms de dossiers est également disponible).

### Audit et tâches DE/NDE / RR/RH

La pipeline accepte `task="de_nde"` (CrossEntropy, labels entiers) et
`task="rr_rh"` (BCEWithLogits, cibles float `[RR, RH]`). Les forwards renvoient
les logits ; l'évaluation utilise respectivement softmax et sigmoid.
L'inventaire découvre les catégories présentes lorsque `class_names` est absent
ou `null` dans le JSON. Les clés historiques sont les noms `*.fits` ;
`cluster_id` est le nom sans extension. Les copies identiques sont fusionnées,
les contenus divergents et les annotations DE/NDE contradictoires sont refusés.
Les anciens dictionnaires avec `class` restent acceptés à travers un adaptateur.

`candidate_policy="positive"` rend cRR/cRH positifs. Avec `"ignore"`,
`label_mask` vaut zéro uniquement pour les candidats sans confirmation du même
label. La loss, les poids positifs, les métriques et les colonnes `known_RR` /
`known_RH` des prédictions respectent ce masque. Une confirmation prime sur un
candidat. L'ancienne valeur ambiguë `"all"` est refusée explicitement.

**Convention RR/RH :** NDE donne `[0,0]`, RR `[1,0]`, RH `[0,1]`,
RR+RH `[1,1]`. DE sans caractérisation RR/RH est exclu. NDE associé à DE,
RR/RH ou un candidat est contradictoire et refusé. Les candidats `ignore`
restent masqués (strates `?0`, `0?`, `??`, etc.), jamais des négatifs confirmés.
Le train doit contenir des positifs et négatifs connus pour chaque label.
L'inventaire doit inclure NDE : utiliser `class_names: null` pour découvrir
les dossiers, ou inclure explicitement NDE dans la sélection des fichiers.

Les splits regroupent les annotations par identifiant avant augmentation.
Les quatre combinaisons sont stratifiées aux deux séparations et dans les folds.
Si leurs effectifs sont insuffisants, `iterative-stratification>=0.1.9` répartit
les états positifs, négatifs et inconnus ; aucun repli aléatoire RR/RH n'est utilisé.
Les effectifs très faibles ne permettent pas de garantir toutes les classes
partout. `split_distribution.json` compte les amas uniques et indique si
chaque ROC-AUC est définie. Le mode pair RR/RH stratifie sa propre population
CHANDRA ; les comparaisons exigent des cohortes identiques. Les poids utilisent
uniquement les annotations connues du train. Le meilleur checkpoint est choisi
sur validation ; les seuils valent 0,5 par défaut et peuvent être sélectionnés
sur validation uniquement. Le test est évalué après restauration et fixation des seuils. La loss CE pondérée de l'époque est maintenant agrégée selon sa
normalisation globale, indépendamment du découpage des lots.

Les SSN utilisent automatiquement un précalcul des 24 variantes normalisées et
augmentées. `model_params` expose `J`, `L`, `max_order` ; le cache se trouve dans
`processed_dir/scattering` (ou `processed/scattering` si `processed_dir=null`).
MixUp est désactivé pour ces modèles. Le cache ne dépend ni des labels ni de la
tâche. Les signatures sources utilisent chemin, taille, mtime et ctime ; un
checksum SHA-256 valide le fichier de coefficients. La version scattering 2
reconstruit les anciens caches ; les FITS existants dont la provenance ne
comprend pas ctime sont également reconstruits, sans modifier les sources.

Pour exécuter les tests depuis le répertoire parent :

```bash
.venv/bin/python -m pip install -e './clusterprep[scattering,test,download]'
PYTHONPATH=clusterprep/src MPLBACKEND=Agg .venv/bin/python -m pytest clusterprep/tests -q
PYTHONPATH=clusterprep/src MPLBACKEND=Agg .venv/bin/python clusterprep/examples/audit_smoke.py results/nouvel_audit
```

`audit_smoke.py` refuse un dossier déjà existant, crée 60 amas synthétiques,
exécute les quatre commandes CLI et conserve quatre entraînements de deux
époques avec leurs artefacts. Voir `AUDIT_REPORT.md` pour les résultats et les
limites de la validation.

### Notebook d'analyse multi-label

Le notebook [`notebooks/analyse_multilabel.ipynb`](notebooks/analyse_multilabel.ipynb)
contrôle les artefacts RR/RH, affiche les scores et courbes par label, choisit
optionnellement les seuils sur validation et compare les mêmes cohortes test
avec bootstrap de clusters. Il inclut un mode synthétique autonome.

Voir le [guide de lancement et contrat d'entrée](notebooks/README_multilabel.md)
pour les nouvelles options `thresholds`, `threshold_selection`, les anciens
schémas supportés et les limites scientifiques des annotations.
# Transfer learning DE/NDE → RR/RH

Le transfert multi-phase utilise les API d'entraînement existantes. Voir le
[guide de transfert](TRANSFER_LEARNING.md), la [configuration ImageCNN](examples/transfer_image_cnn.json)
et le [pilote comparatif](examples/transfer_pilot.py).
