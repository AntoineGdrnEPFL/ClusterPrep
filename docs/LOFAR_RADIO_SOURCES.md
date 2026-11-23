# Sélection du produit radio LOFAR

Dans une expérience JSON, `radio_source` sélectionne `RAW` (défaut), `T25kpc`,
`T50kpc`, `T100kpcSUB`, etc. `mode` et les modalités logiques restent inchangés.
L'[exemple de comparaison](../configs/lofar_radio_sources/comparison.json) contient
deux expériences identiques sauf `radio_source` ; les noms de répertoires des runs
restent uniques. Les chemins de cette suite sont relatifs au répertoire de lancement.

```json
{"name": "cnn_rr_rh_t25kpc", "mode": "raw", "model": "cnn",
 "task": "rr_rh", "radio_source": "T25kpc", "missing_radio_policy": "error"}
```

`missing_radio_policy="error"` est la seule politique supportée : tous les amas de
l'inventaire fourni doivent disposer du produit. Le message donne les IDs manquants.
Aucun remplacement par RAW, aucune suppression ni modification des folds n'a lieu.
Les noms sont sensibles à la casse, sans séparateurs de chemins, et vérifiés contre
les produits découverts. La découverte reconnaît RAW, T… (y compris SUB), Blur… et
R-… ; CHANDRA, XMM, optique et compact-model sont exclus.

## Inventaire et convention vérifiée

Les annotations et IDs proviennent exclusivement de `RAW/<annotation>/*.fits`.
Les copies dans plusieurs catégories fusionnent les labels RAW et sont comparées
par SHA-256. Des contenus différents pour un même amas/produit font échouer
l'inventaire. Les catégories tapered ne créent aucun label.

Les noms acceptés sont `<ID>.fits`, `<ID><produit>.fits`,
`<ID>_<produit>.fits` et `<ID>-<produit>.fits`. L'association doit correspondre à
un ID RAW exact ; toute ambiguïté provoque une erreur. Cette convention a été
vérifiée sur les fichiers locaux, notamment :

- `PSZ2G112.48+56.99.fits`
- `PSZ2G112.48+56.99T25kpc.fits`
- `PSZ2G112.48+56.99T50kpc.fits`
- `PSZ2G112.48+56.99T25kpcSUB.fits`

L'[audit réel du 8 novembre 2026](lofar_radio_inventory.json), produit depuis
`scratch/data/PSZ2/classified`, comporte 236 amas canoniques (six catégories
historiques). RAW et R-1.25 couvrent les 236 ; T25kpc, T50kpc et T100kpc, avec leurs
variantes SUB, en couvrent 209, laissant 27 absents. Les produits angulaires couvrent
les 27 autres. Aucun nom incompatible ni doublon contradictoire n'a été trouvé.
Les orphelins sont relatifs à cet inventaire : ils incluent les fichiers de catégories
non retenues, notamment cDE ; ils ne prouvent pas une erreur dans le survey.
Les listes complètes des présents, manquants, copies identiques et orphelins sont
conservées dans l'audit.

```python
from clusterprep.io import get_data_info_dict
info, audit = get_data_info_dict("scratch/data/PSZ2/classified", return_audit=True)
```

L'inventaire doit lire les FITS pour vérifier leurs contenus ; le catalogue local
représente environ 37 Go. Le CLI exporte l'audit et chaque run enregistre sa couverture.

## Comparer sur un sous-ensemble commun explicite

Le catalogue local complet ne peut donc pas lancer une comparaison RAW/T25kpc
stricte sans choisir préalablement une population commune. L'exemple JSON conserve
volontairement `error`. Pour utiliser les 209 amas communs, documenter la sélection
**avant** de construire un seul jeu de splits partagé :

```python
import json
from pathlib import Path
from clusterprep.io import get_data_info_dict
from clusterprep.training import ExperimentConfig, run_experiments

spec = json.loads(Path("clusterprep/configs/lofar_radio_sources/comparison.json").read_text())
info, audit = get_data_info_dict(spec["data_dir"], spec["class_names"], return_audit=True)
sources = [c["radio_source"] for c in spec["experiments"]]
common = {k: v for k, v in info.items() if all(s in v["radio_paths"] for s in sources)}
output = Path(spec["output_dir"])
output.mkdir(parents=True, exist_ok=True)
(output / "common_population.json").write_text(json.dumps({
    "sources": sources, "included": sorted(common),
    "excluded": sorted(set(info) - set(common)), "inventory": audit
}, indent=2))
runs = run_experiments(common, [ExperimentConfig(**c) for c in spec["experiments"]],
                       output, class_names=spec["class_names"], **spec["split"])
```

Le nombre commun dépend des catégories choisies. Les splits sont définis par les
IDs/labels RAW ; le chemin radio actif n'est pas l'identité de l'amas. La comparaison
existante contrôle la cohorte test, et la provenance du FITS actif est enregistrée
séparément dans `splits.json`. La CV reçoit la même vue du produit et réajuste les
statistiques dans chaque fold. Le transfert de poids compatible conserve le produit
de la cible et consigne les domaines source/cible.

## WCS, normalisation et caches

Chaque produit définit sa propre grille céleste, son crop centré et son facteur de
réduction (plus grande puissance de deux permettant `size × size`). CHANDRA et
l'optique sont reprojetés sur cette grille, puis les images et masques subissent les
mêmes transformations. Une carte plus petite que `size` échoue explicitement ; pas
de padding ni de changement d'échelle implicite. Le champ de vue angulaire peut
**différer entre produits** : comparer à taille identique ne garantit pas un champ
physique identique.

Pour l'amas vérifié, les formes RAW/T25kpc/T50kpc sont respectivement 3264²,
1332² et 666², avec des pixels de 1,5 / 3 / 7 secondes d'arc. Les faisceaux diffèrent,
et l'unité observée est JY/BEAM. Les caches conservent BMAJ, BMIN, BPA et BUNIT ; le
bruit utilise le faisceau du produit et la WCS réduite, y compris lorsqu'il doit
relire les métadonnées manquantes d'un cache ancien. Aucun lissage vers un faisceau
commun ni conversion d'unité n'est ajouté. Le scattering conserve son comportement
pixel et opère après la normalisation existante.

La normalisation historique par image (percentiles 30/99 et asinh) est préservée.
Avec modalités explicites et `normalization.radio.scope="train"`, les bornes sont
réajustées uniquement sur le train du run/fold, puis utilisées en validation/test.
Les bornes d'un entraînement RAW ne sont pas reprises lors d'un entraînement tapered.
Les cartes SUB observées ont des valeurs négatives, conservées dans les caches
physiques ; la normalisation historique applique ensuite son clipping percentile.
Aucune interprétation physique supplémentaire de la soustraction SUB n'est supposée.

Les chemins et identités des caches RAW v2 restent inchangés. Les autres produits
utilisent `processed/<produit>/<ID>/…` (et un sous-espace analogue pour le scattering).
Le marqueur `_RAW_` dans les noms FITS internes désigne le canal historique ; le
répertoire produit et la provenance désignent sa source réelle. Les clés incluent
produit, version de prétraitement, chemin, taille, mtime et ctime du FITS actif.
Les caches CHANDRA et optiques sont eux aussi séparés car leur grille dépend du produit.
Aucun cache historique n'est déplacé ou réutilisé pour un autre produit.

Le CLI de prétraitement accepte `--radio-source T25kpc`. L'API expose
`select_radio_source(info, "T25kpc")` et `resolve_radio_path(item)` ;
`ClusterDataset(..., radio_source="T25kpc")` permet également une sélection directe.
`raw_path` reste toujours RAW. Les galeries issues des runs tapered choisissent
leurs caches radio/CHANDRA spécifiques ; `processed_root` désigne la racine commune
des caches. Les exports incluent `radio_source`.

## Vérifications

Les tests dédiés sont dans [test_radio_source.py](../tests/test_radio_source.py) :
non-régression RAW, association/ambiguïtés/doublons/orphelins, absence stricte,
WCS/masque/faisceau, cache et invalidation, modes historiques et explicites,
reprojection optique, normalisation train, galeries, splits/CV, transfert des poids,
et une époque d'entraînement suivie d'une inférence pour RAW et T25kpc.

Validation exécutée le 8 novembre 2026 : la suite complète donne **497 tests
réussis, 1 ignoré et 6 échecs préexistants** (33 sous-tests réussis). Ces six
échecs ont été reproduits avec une copie des sources antérieures à cette
implémentation : deux exemples JSON rr_binary/rh_binary absents, deux tests de
notebooks, et deux attentes de présentation graphique (nombre d'axes/langue des
titres). Ils ne sont pas modifiés par ce travail. Le test supplémentaire de CV
T25kpc vérifie aussi les deux folds, l'absence du test externe dans ceux-ci et le
réentraînement final.

Un [smoke test sur les FITS réels](lofar_real_smoke.json) a également entraîné le
même CNN une époque sur RAW puis T25kpc, sur les mêmes 12 amas (6 train, 2 validation,
4 test). Les deux inférences ont réussi et l'identité de cohorte test est identique.
Ces runs servent à valider l'exécution ; leurs métriques ne constituent pas une
comparaison scientifique des produits.
