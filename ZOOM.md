# Zoom indexé

Le zoom est optionnel dans chaque entrée `experiments` des JSON existants :

```json
{
  "name": "zoom-moderate",
  "augment": true,
  "augmentation": {
    "zoom": {
      "enabled": true,
      "factors": [0.9, 1.0, 1.1],
      "interpolation": "bilinear",
      "padding_mode": "constant",
      "fill_value": 0.0
    }
  }
}
```

`factors` est obligatoire lorsque le zoom est activé : liste non vide de nombres
finis strictement positifs, sans tirage aléatoire. Les autres paramètres ont les
valeurs par défaut ci-dessus. `interpolation` accepte `bilinear` ou `nearest` ;
`padding_mode` accepte `constant`, `reflect`, `replicate`. `fill_value` est une
valeur finie dans l'espace normalisé, utilisée uniquement pour le padding
constant radio. Aucun bruit n'est ajouté implicitement.

Sans cette section, avec `enabled: false`, ou avec `augment: false`, le
comportement antérieur est conservé. Le runner active l'augmentation uniquement
sur le train, après les splits par amas, y compris dans les folds internes de CV.
L'API directe conserve son commutateur historique `ClusterDataset(augment=True)` :
ne pas l'activer pour un dataset d'évaluation.

## Géométrie et indexation

Les 24 rotations/flips restent dans leur ordre historique, répétées pour chaque
facteur dans l'ordre du JSON. Pour `N = 24 * len(factors)` :

```python
source_index, variant_index = divmod(index, N)
zoom_index, geometry_index = divmod(variant_index, 24)
rotation_index, flip_index = divmod(geometry_index, 2)
```

`dataset.N_VARIANTS` donne le nombre de variantes configurées ;
`dataset.get_item_info(index)` expose l'amas source, la rotation, le flip et
`zoom_factor`. Aucun amas augmenté n'est matérialisé en mémoire : le cache image
conserve uniquement les originaux normalisés, et la transformation est calculée
à la demande.

L'ordre reste : alignement WCS → crop/réduction → normalisation par image
(percentiles 30–99 et asinh) → géométrie commune → entrée modèle/scattering.
Il n'y a pas de statistiques globales apprises sur le train dans cette pipeline.
Le zoom ne recalcule pas les percentiles : les bornes de l'image source restent
identiques pour toutes ses variantes, également avec le bruit radio explicite.

Le centre est `(width - 1)/2, (height - 1)/2`. Un facteur supérieur à 1 agrandit
les structures ; un facteur inférieur à 1 réduit le champ observé dans l'image.
Rotation, flip et zoom utilisent une seule grille affine, sans redimensionnements
intermédiaires. Pour **le facteur 1**, on conserve exactement la rotation/flip
historique en nearest-neighbor et son remplissage zéro, quels que soient les
paramètres d'interpolation/padding du zoom. Cela permet une comparaison exacte
avec les anciennes variantes ; les variantes avec un zoom différent de 1
utilisent les nouvelles options.

Radio, CHANDRA et masque partagent cette grille. CHANDRA est interpolé en
séparant les intensités valides et les poids de couverture, puis normalisé par
les poids (> 1e-6). Le masque est transformé en nearest-neighbor et reste binaire.
Le padding de CHANDRA est toujours invalide, même si la radio utilise réflexion
ou prolongement des bords. CHANDRA absent reste nul avec un masque nul ; les
NaN ne sont jamais interpolés directement. Les modes existants `raw`, `pair`
et `multimodal` et les tâches binaires/multilabel utilisent le même mécanisme.

## Scattering et caches

Les modèles calculant le scattering en ligne reçoivent directement les images
transformées. Pour `dual_ssn`, le précalcul utilise exactement le même dataset.
Le cache inclut les facteurs **dans leur ordre**, interpolation, padding,
remplissage et version de géométrie dans ses métadonnées et son nom haché.
Changer un paramètre produit un autre cache, sans écraser le précédent.
Les coefficients ne sont jamais zoomés après leur calcul.

Les caches historiques sans zoom restent réutilisables. Le runner prépare un
cache zoomé pour le train et des caches historiques pour validation/test :
l'évaluation utilise toujours une image identité, même si `factors` ne contient
pas 1. Pour le précalcul manuel, passer le même dictionnaire `zoom=...` à
`prepare_scattering`, `load_scattering` et `ClusterDataset`. Préparer aussi le
cache sans zoom si ce même amas doit être évalué. Le cache FITS prétraité reste
commun, car il précède toute augmentation. La restriction existante sur le bruit
radio aléatoire avec scattering précalculé reste applicable.

## Visualiser un amas

```python
from clusterprep import ClusterDataset, get_data_info_dict, plot_zoom_variants

info = get_data_info_dict("scratch/data/PSZ2/classified")
key = next(k for k, v in info.items() if v.get("chandra_path"))
ds = ClusterDataset(info, [key], [], mode="multimodal", include_targets=False)
fig = plot_zoom_variants(ds, factors=[0.9, 1.0, 1.1, 1.2],
                         output_path="results/zoom-preview.png")
```

La figure présente les facteurs en colonnes et radio/CHANDRA/masque en lignes,
avec une échelle de couleurs commune par modalité. `geometry_index=0..23`
permet d'inspecter aussi les rotations/flips. Les options de zoom du dataset
sont reprises, avec une surcharge possible par `zoom={...}`. Le diagnostic
utilise une seule image propre, sans bruit radio aléatoire ni scattering.

## Comparaisons A/B/C

Deux suites sont fournies : [DE/NDE](configs/zoom/de_nde.json) et
[RR/RH multilabel](configs/zoom/rr_rh.json). Elles conservent architecture,
hyperparamètres, seed et découpage communs pour A/B/C :

| Expérience | Facteurs | Variantes/amas | Exemples/époque vs A |
|---|---|---:|---:|
| A — baseline | zoom désactivé | 24 | ×1 |
| B — modéré | 0.9, 1.0, 1.1 | 72 | ×3 |
| C — étendu | 0.8, 0.9, 1.0, 1.1, 1.2 | 120 | ×5 |

Depuis la racine du projet, lancer par exemple :

```sh
PYTHONPATH=clusterprep/src .venv/bin/python -m clusterprep clusterprep/configs/zoom/rr_rh.json
```

Les suites sélectionnent initialement `fold_index: 0` sur cinq folds. Répéter
avec `fold_index` de 0 à 4, en conservant `split_seed: 42`, et nommer chaque
expérience avec le suffixe `_fold_0`…`_fold_4` pour l'analyse agrégée existante.
Comparer F1 macro, F1 RR/RH, PR-AUC par label, ROC-AUC, loss de validation et
moyenne/écart-type inter-folds avec les sorties habituelles de
[la pipeline d'analyse](UNIFIED_ANALYSIS.md). Les scores RR/RH par label
concernent la suite multilabel. Les fichiers `splits.json` permettent de vérifier
que les populations comparées sont identiques.

À batch size constant, les nombres de mises à jour par époque augmentent
approximativement de ×3 et ×5 (arrondis et batching inclus). Une comparaison à
nombre d'époques égal ne contrôle donc pas le budget d'optimisation. Rapporter
les époques effectivement exécutées et le nombre de batches (enregistré dans
`config.json → augmentation_summary`, avec le nombre d'exemples par split et
de variantes par amas) ; ajouter si besoin
une comparaison à budget d'updates comparable. Les configurations ne constituent
pas des résultats expérimentaux : aucune amélioration scientifique n'est présumée.

Commencer avec B et inspecter plusieurs amas, notamment les RR périphériques.
Le zoom-in peut couper des structures responsables des labels. Le zoom numérique
modifie aussi le beam et les corrélations du bruit ; il ne simule pas physiquement
un changement de redshift. Le zéro radio correspond au bas de l'échelle normalisée
[0, 1], et peut créer une bordure détectable : comparer les paddings sur validation.
