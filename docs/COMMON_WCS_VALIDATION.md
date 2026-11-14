# Validation d’une grille WCS commune

Ouvrir [le notebook](../notebooks/validate_common_wcs_preprocessing.ipynb) avec
l’environnement du projet (`pip install -e '.[analysis,test]'` si nécessaire).
Exécuter ses cellules dans l’ordre. Il n’importe aucun cache ni runner d’entraînement.
Les nouvelles fonctions sont dans [common_wcs_validation.py](../src/clusterprep/common_wcs_validation.py).

## Choix du périmètre

- `CLUSTER_IDS = "PSZ2G073.97-27.82"` : exemple rapide.
- `CLUSTER_IDS = ["PSZ2G073.97-27.82", "PSZ2G052.08+46.13"]` : sélection, dont fallback.
- `CLUSTER_IDS = None` : catalogue officiel complet, y compris amas sans FITS locaux.

Le catalogue officiel est téléchargé une fois dans le nouveau dossier `inputs/`.
Le téléchargement initial nécessite internet ; les reprises utilisent le snapshot.
Le [catalogue LoTSS](https://lofar-surveys.org/planck_dr2.html) indique RA/Dec en degrés,
R500 en kpc et M500 en 10¹⁴ masses solaires. Citer Botteon et al. (Paper I) lors
 de l’utilisation des produits, ainsi que les travaux X-ray pertinents.
Un catalogue alternatif FITS/ECSV doit porter ces colonnes et des unités explicites.
Le centre est celui du catalogue, interprété en ICRS/J2000, jamais celui d’un FITS.

## Géométrie et limites

Planck18, grille 128², nord en haut, est à gauche, projection TAN. Le côté nominal
est 4 R500 / DA(z) dans le plan de projection (approximation petit angle).
Sans R500, M500 et z permettent R500 = [3 M500 / (4π 500 ρcrit(z))]^(1/3).
Sinon la médiane des champs valides du catalogue entier est calculée une fois par
run. Elle est indépendante de la sélection d’amas à traiter. Pour un futur
entraînement, calculer cette médiane uniquement sur le train du fold, ou la fixer
avec une référence externe indépendante.

Reprojection bilinéaire directe sur cette grille, sans égalisation des beams,
conversion photométrique ni conservation générale du flux intégré en Jy/beam.
Ce choix peut sous-échantillonner les sources compactes : comparer les géométries
avant de décider d’un filtre anti-aliasing ou d’une méthode photométrique.
Les beams enregistrés restent les beams natifs. Aucun calcul ne modifie CDELT/PC/CD
des entrées : Reproject utilise le WCS source complet et une cible neuve.

La normalisation appelle la fonction existante `normalize`, P30/P99/asinh(alpha=10),
par image/bande. Les politiques train/globales ne sont pas activées. Avant
normalisation, la visualisation utilise un asinh P1–P99 propre à chaque panneau :
les couleurs ne représentent pas une échelle physique partagée entre modalités.
Legacy est affiché en bandes individuelles g/r/z, avec indication des absences.

## Sorties

Dans `outputs/common_wcs_validation/` par défaut :

- `figures/<id>_common_wcs.png` : une figure par amas avec grille valide, sans affichage inline.
- `processed/<id>/*.fits` : images physiques et masques binaires pour chaque modalité/bande.
- `processed/<id>/metadata.json` : coordonnées, champ, WCS, propriétés physiques,
  sources, erreurs, absences, couvertures par bande et identité de reprise.
- `summary.csv` : une ligne par amas sélectionné, diagnostics et reprise.
- `configuration.json` : paramètres, médiane et taille de sa population,
  cosmologie, unités, provenance du catalogue et versions/code.

`SAVE_FIGURES`, `SAVE_PROCESSED` et `VIEW` contrôlent les exports. Les métadonnées et
le résumé restent écrits pour la traçabilité. Les NaN ne deviennent pas zéro dans
les FITS ; leurs masques préservent les zéros physiques. `has_*` indique l’existence
du fichier, pas sa lisibilité. Les erreurs d’une modalité n’empêchent pas les autres.
La couverture Legacy du CSV est la moyenne des bandes produites, pas des bandes absentes.

`OVERWRITE=False` réutilise seulement les résultats réussis dont configuration,
contenu des sources et contenu des sorties sont inchangés. Les cas incomplets,
modifiés ou corrompus sont recalculés. `OVERWRITE=True` force tout le périmètre
sélectionné. Un fichier source ajouté est aussi détecté. Les anciens FITS devenus
obsolètes dans le dossier propre à l’amas sont supprimés lors d’un recalcul réussi.
Le résumé et la configuration décrivent la sélection du dernier run ; les autres
amas déjà exportés conservent leur propre manifeste. Ne pas exécuter deux runs
simultanément dans le même dossier. Un dossier de sortie non vide et non identifié
comme appartenant au validateur est refusé (hors son sous-dossier `inputs/`).

## Tests

Depuis la racine du projet :

```bash
PYTHONPATH=clusterprep/src .venv/bin/python -m pytest clusterprep/tests/test_common_wcs_validation.py -q
```

Les tests couvrent les coordonnées réellement reprojetées en CDELT/PC/CD,
la grille commune, l’indépendance X-ray/Legacy envers la présence des radios,
les masques, le fallback, les unités, l’export/relecture et l’invalidation des reprises.
