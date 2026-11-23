# Préparation radio appariée — phases 1 à 4

API physique indépendante et intégration optionnelle dans `ClusterDataset`, sans activation implicite. Voir [audit](MATCHED_RESOLUTION_AUDIT.md) et [résultats de validation](../../outputs/matched_resolution_phase123/validation_report.md).

```python
from clusterprep.matched_resolution import prepare_matched_pair

pair = prepare_matched_pair(
    raw_path, t50kcpsub_path, chandra_path,
    record={"cluster_id": "example", "ra_deg": 170., "dec_deg": 30.,
            "redshift": .2, "r500_kpc": 1000.},
    size=128, pixels_per_beam=3., min_support=.999,
    output_unit="Jy/beam",  # beam cible, ou "Jy/sr"
)
tensors = pair.tensors()
# tensors['raw'], tensors['t50kcpsub'], tensors['chandra'] : float32 [1,128,128]
# trois masques booléens : *_mask ; aucune normalisation
# partager EXACTEMENT tensors['chandra'] et tensors['chandra_mask'] entre A et B
common_mask = pair.common_radio_mask
metadata = pair.metadata
```

Pour des images déjà chargées, utiliser `RadioPlane` et `match_pair`. Le CHANDRA fourni à cette dernière doit déjà être sur la WCS de sortie ; il est conservé exactement. `read_radio` accepte un masque de couverture externe sur la grille native ; sinon seuls les pixels finis sont présumés observés. Les zéros finis ne sont jamais considérés automatiquement comme absents. Les cubes avec axes non singleton sont refusés par le lecteur radio.

Conventions : côté 4R500, centre catalogue, Planck18, TAN, cible circulaire à partir des deux beams et de la Jacobienne de sortie. Convolution native en brillance de surface avant reprojection. Les trous restent invalides. Les données FITS conservent les NaN, les tenseurs les remplacent explicitement par zéro avec masque. L'API ne réalise aucune sélection d'amas ni de normalisation.

La cible est une PSF nominale ; l'interpolation ajoute un faible écart mesuré dans les tests. Noyaux nécessitant une déconvolution, unités ambiguës, WCS variant de plus de 1 % sur le maillage de contrôle, noyaux de plus de 2049 pixels de côté et erreur de covariance discrète >3 % sont refusés explicitement. Le support minimal 0,999 et la troncature à 5σ évitent d'inventer des observations aux bords, mais la photométrie n'est garantie que loin des régions invalides. Le champ 4R500 n'est jamais réduit pour masquer une absence d'observations.

Exemple réel (exige les FITS locaux et le catalogue déjà utilisé pour la validation) :

```sh
.venv/bin/python clusterprep/examples/validate_matched_resolution.py --output outputs/my_matched_run
```

Le répertoire de sortie doit être nouveau. L'exemple exporte les FITS, masques, tenseurs, métadonnées, couvertures et galeries pour trois amas. Il n'utilise aucun cache d'entraînement. La phase 4 est décrite ci-dessous ; les phases 5–7 restent hors périmètre.

## Phase 4 — intégration optionnelle dans ClusterDataset

Le constructeur conserve `radio_preprocessing="historical"` par défaut. Les scripts d'entraînement et leurs configurations ne sont pas modifiés. L'activation se fait explicitement lors de la création du dataset :

```python
from clusterprep import ClusterDataset

options = {
    "radio_mask": "intersection",  # défaut ; autre choix : "specific"
    "target_pixels_per_beam": 3.0,
    "min_support": 0.999,
    "truncate": 5.0,
    "output_unit": "Jy/beam",
    # Valeurs illustratives, à fixer scientifiquement pour la future cohorte :
    "coverage_thresholds": {"RAW": 0.8, "T50kpcSUB": 0.8,
                            "CHANDRA": 0.2, "intersection": 0.8},
    "coverage_policy": "error",  # "report" permet l'inspection sans rejet
}
common = dict(
    mode="pair", task="rr_rh", size=128,
    radio_preprocessing="matched_resolution",
    matched_resolution=options,
    matched_records=records_by_cluster_id,
    processed_dir="processed", cache=True,
)
a = ClusterDataset(data_info, existing_keys, ["RR", "RH"], radio_source="RAW", **common)
b = ClusterDataset(data_info, existing_keys, ["RR", "RH"], radio_source="T50kpcSUB", **common)
metadata = a.get_resolution_metadata(0)
# Ne lance aucun entraînement et ne génère/modifie aucun split.
```

Chaque entrée de `data_info` conserve `raw_path` pour RAW et fournit `radio_paths["T50kpcSUB"]`, ainsi que `chandra_path` si disponible. Les métadonnées physiques proviennent de `matched_records[cluster_id]` ou de `info["physical_record"]` : `ra_deg`, `dec_deg`, `redshift`, `r500_kpc` (ou `m500_msun`). Aucun accès réseau ni lecture implicite d'un catalogue global. Les unités sont degrés, kpc et masses solaires ; les identifiants du record et du dataset doivent correspondre.

### Masques, couvertures et éligibilité

- `intersection` applique `raw_mask & t50kcpsub_mask` aux **deux radios**, avant normalisation et augmentations. CHANDRA garde son propre masque.
- `specific` applique à chaque radio son propre masque. Les fractions originales restent conservées dans les deux cas.
- `coverage` contient `RAW`, `T50kpcSUB`, `CHANDRA`, `intersection` sur le champ commun **avant augmentation** ; `selected_radio_coverage` décrit le masque radio choisi.
- `coverage_thresholds` est vide par défaut : aucune coupure scientifique implicite. Les seuils sont inclusifs (`fraction >= seuil`) et évalués pour toute la paire, indépendamment du produit sélectionné. `coverage_policy="error"` lève `CoverageEligibilityError` si un seuil échoue ; l'exception contient les métadonnées. `report` conserve l'échantillon avec `eligible=False`. Aucune politique ne supprime de clés.
- `get_resolution_metadata(index)` reste accessible pour les amas non éligibles. Il peut déclencher le prétraitement au premier accès. Les erreurs de sources/physique restent explicites ; elles ne deviennent pas des images de remplacement.

`target_resolution` contient `fwhm_arcsec`, `fwhm_kpc`, `fwhm_over_r500`, `r500_kpc` et la cosmologie Planck18. La conversion utilise le redshift et le R500 réel ou dérivé qui définit le champ, pas le rayon nominal d'une autre source.

### Sorties et normalisation

Modes pris en charge : `raw`, `pair`, `multimodal` et modalités explicites composées de `radio`/`xray`. Les tenseurs conservent les noms et formes historiques ; `raw` désigne le produit sélectionné. Le masque `raw_mask` est toujours fourni et les pixels invalides sont nuls dans les tenseurs. Le champ `resolution_metadata` de l'échantillon est un sous-ensemble compatible avec `DataLoader` ; les métadonnées complètes, WCS et chemins sont accessibles par `get_resolution_metadata` et, pour les échantillons annotés, `get_item_info`.

Aucun canal supplémentaire n'est imposé aux modèles : avec les modalités explicites radio+xray, `image` conserve les trois canaux radio, CHANDRA, masque CHANDRA. Le masque radio est fourni séparément. Les percentiles/asinh existants sont appliqués seulement aux pixels valides ; `normalization` et `normalization_bounds` sont honorés. `fit_normalization` utilise le nouveau prétraitement pour les datasets à modalités explicites ; il reste de la responsabilité de l'appelant de lui fournir uniquement les clés train. Aucune statistique globale n'est ajustée automatiquement.

Les rotations, flips et zooms utilisent la même transformation pour chaque image et son masque. CHANDRA et son masque sont strictement identiques entre A et B à options de normalisation, bornes et variante géométrique identiques. Le masque radio n'est jamais appliqué à CHANDRA. Dans `raw` ou `multimodal`, CHANDRA absent produit une couverture nulle ; `pair` et les modalités déclarées obligatoires continuent à exiger sa présence. Le mode harmonisé signale une modalité obligatoire absente au lieu de filtrer silencieusement les clés.

### Cache apparié distinct

Emplacement par défaut : `processed/matched_resolution/v1/<cluster_id>/<sha256>.npz`. `matched_cache_dir` remplace la **racine** de stockage (le sous-répertoire `matched_resolution/v1` y est toujours ajouté). `processed_dir=None` désactive le disque, sauf racine explicite `matched_cache_dir`. `cache=True` active aussi la rétention mémoire des paires.

Une archive contient les deux radios, CHANDRA une seule fois, leurs trois masques, les métadonnées et l'identité JSON. Les pixels sont physiques, sans normalisation ni augmentation. La clé dépend :

- des chemins, tailles et SHA256 des deux FITS radio et du FITS CHANDRA éventuel ;
- du beam cible déterminé **après lecture des deux beams natifs**, de la grille WCS, de la taille et du record physique ;
- des paramètres de convolution/échantillonnage/unités, des versions du pipeline/cache, des empreintes des modules physiques et des bibliothèques numériques.

La sélection RAW/T50kpcSUB, le choix de masque et les seuils sont des politiques de lecture : ils réutilisent la même paire physique sans reconvolution, mais sont enregistrés dans les métadonnées de la vue. Les labels, splits et paramètres d'entraînement ne sont pas des entrées de prétraitement.

Les empreintes sources sont mémorisées dans le processus tant que taille/mtime/ctime/inode restent inchangés ; un nouveau processus rehash les fichiers. Elles sont contrôlées même pour les hits mémoire. Écriture atomique, CRC ZIP et SHA256 des tableaux, reconstruction du cache corrompu, refus de publier si une source change pendant le calcul. Les sources doivent rester disponibles. Des accès concurrents peuvent dupliquer le calcul initial, mais ne publient pas d'archive partielle. Le cache historique FITS et le cache scattering ne sont jamais réutilisés ou écrasés.

### Limites explicites et validation

Le mode harmonisé refuse pour l'instant le scattering précalculé historique, l'augmentation `radio_noise` et les modalités optiques `legacy` : leurs grilles/résolutions/caches nécessitent une adaptation dédiée. Ces fonctionnalités restent inchangées en mode historique. La grille reste celle de côté 4R500, sans fallback physique implicite, avec les limites numériques de PSF des phases 1–3.

Exemple de vérification réelle sans entraînement :

```sh
.venv/bin/python clusterprep/examples/validate_matched_dataset.py --output outputs/my_phase4_validation
```

Le répertoire doit être nouveau. Le script compare trois amas, crée un cache distinct et exporte les métadonnées des deux vues, les couvertures, les identités CHANDRA et l'éligibilité selon des seuils **illustratifs**. Voir le [rapport de phase 4](../../outputs/matched_resolution_phase4/implementation_report.md).
