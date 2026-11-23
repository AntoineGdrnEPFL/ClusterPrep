# Diagnostic de résolution et bruit radio

Le [module](../src/clusterprep/radio_resolution_diagnostics.py) lit le résumé et les
métadonnées du validateur commun, puis les FITS natifs référencés. Il ne modifie
ni les images, ni le preprocessing, ni les modèles et configurations d’entraînement.

## Exécution

Depuis la racine du projet :

```bash
PYTHONPATH=clusterprep/src .venv/bin/python -m clusterprep.radio_resolution_diagnostics \
  --config clusterprep/configs/diagnostics/radio_resolution_noise.json \
  --project-root .
```

Ou exécuter le [notebook](../notebooks/diagnose_radio_resolution_noise.ipynb).
L’environnement `clusterprep[analysis,test]` fournit les dépendances d’analyse.
Les chemins du [JSON](../configs/diagnostics/radio_resolution_noise.json) sont
relatifs à `project-root`, sauf s’ils sont absolus. `cluster_ids: null` parcourt
le résumé entier ; une liste limite le diagnostic. Les seuils configurables par
défaut sont couverture <0,8 et différence absolue de couverture ≥0,1. Ces seuils
sont des avertissements, jamais des exclusions.

Les résultats vont uniquement dans `outputs/radio_resolution_diagnostic/` :

- `metrics.csv` : 1 ligne par amas/modalité, y compris les absences et erreurs ;
- `pairs.csv` : comparaisons appariées, unités, rapports et différences ;
- `metadata_issues.csv`, `coverage_issues.csv`, `sampling_discordance.csv` : cas à examiner ;
- `report.md` : résultats mesurés, limites et recommandations A/B/C ;
- `figures/` : six figures statistiques et une galerie déterministe ;
- `configuration.json`, `checkpoints/` : provenance et reprise par produit.

Les checkpoints ne sont réutilisés que si les empreintes du code, de la source,
des métadonnées et de la ligne du résumé concordent. Les sources absentes sont
également mises en cache, mais leur apparition invalide le checkpoint. Les erreurs
inchangées sont conservées ; `overwrite: true` force leur nouvelle mesure. Les
figures et le rapport sont régénérés sans recalculer les RMS. Éviter les exécutions
concurrentes dans le même dossier. Le diagnostic vérifie aussi si la source native
correspond encore à celle ayant produit les couvertures de validation.

## Lecture des métriques

- `pixels_per_beam_minor/major` : axes principaux de la covariance du beam en
  coordonnées pixels ; pour une grille isotrope, BMIN/Δθ et BMAJ/Δθ.
- `pixels_per_beam_*_direction` : longueur du vecteur céleste BMIN/BMAJ projeté
  dans le plan pixel ; peut différer des axes principaux avec anisotropie.
- `pixel_scale_x/y_arcsec` : normes des colonnes de la Jacobienne locale est/nord.
- `pixel_scale_arcsec` : échelle équivalente √|det J| ; `pixel_anisotropy` compare
  les valeurs singulières et détecte aussi un cisaillement.
- `beams_width_over_bmaj/bmin` : compte **linéaire**, largeur céleste / FWHM.
- `beams_width_x/y_projected` : largeur/hauteur en pixels divisée par l’étendue
  projetée du beam sur cet axe, tenant compte de BPA.
- `beams_area` : aire sphérique du champ / [π BMAJ BMIN / (4 ln 2)].
- `covered_beams_area_approx` : ce compte multiplié par la couverture, approximation
  utilisant la fraction de pixels plutôt qu’une carte des aires individuelles.
- `required_size_for_3pix` : taille carrée hypothétique, au même FoV, pour que la
  plus courte FWHM ait 3 pixels (mise à l’échelle isotrope de la grille).
- `rms_native` : MAD convertie en σ, dans les unités originales de l’image native.
- `rms_canonical` : conversion explicite vers Jy/beam, Jy/sr, Jy/pix ou K.
  Les rapports sont limités à une même famille d’unités ; les préfixes restent
  sensibles à la casse (`mJy` n’est pas `MJy`). Pas de conversion de surface de beam.
- `background_beams_approx` : aire des pixels natifs retenus / aire beam ; ce n’est
  pas un nombre de degrés de liberté statistiquement établi.

Les seuils `<1`, `[1,2[`, `[2,3[`, `≥3` sont indicatifs, pas des garanties. Le beam
est celui du header natif : la PSF effective après interpolation n’est pas mesurée.
Le compte de beams natifs n’indique pas combien restent représentables à 128².

## Bruit : algorithme et limites

Réutilise directement [background_mask](../src/clusterprep/radio_noise.py) sur les
valeurs natives. Pixels finis hors rectangle central de demi-largeur/demi-hauteur ;
jusqu’à 8 itérations médiane/MAD, rejet symétrique à 3σ, σ=1,4826022185 MAD ; au moins
32 pixels. Pas de segmentation externe, de dilatation des sources ni de RMS
calculé après interpolation. La médiane de fond et la fraction retenue sont exportées.

Le bruit est corrélé spatialement, peut varier dans le champ, et les résidus de
soustraction ainsi que les ailes de sources peuvent contaminer le fond. Les fonds
natifs de RAW/T50kpcSUB ne sont pas forcément co-spatiaux. Un rapport de σ en
Jy/beam compare des produits à beams différents, pas une sensibilité identique à
la brillance surfacique ni un effet pur du tapering. Aucun intervalle d’incertitude
reposant sur l’indépendance des pixels n’est calculé.

## Tests

```bash
PYTHONPATH=clusterprep/src .venv/bin/python -m pytest clusterprep/tests/test_radio_resolution_diagnostics.py -q
```

Conversions d’unités, seuils, aire gaussienne, comptes linéaires/en aire, WCS
CDELT/PC/CD, anisotropie/BPA, RMS robuste, sources et métadonnées invalides,
compatibilité des rapports, isolation des écritures et invalidation des reprises.
