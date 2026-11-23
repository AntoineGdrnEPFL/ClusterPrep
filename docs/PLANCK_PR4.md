# Intégration Planck PR4 — audit et utilisation

## Audit préalable et points d’extension

La pipeline existante contient déjà les abstractions nécessaires ; aucune seconde pipeline d’entraînement n’a été créée.

| Responsabilité | Code inspecté / extension |
|---|---|
| FITS et sélection RAW/T50kpcSUB | [io.py](../src/clusterprep/io.py) : `load_celestial_fits`, `resolve_radio_path`, `select_radio_source`, inchangés |
| Reprojection X-ray, crop, réduction | [preprocessing.py](../src/clusterprep/preprocessing.py) : `preprocess`, `centered_raw_window`, `block_average`, inchangés et réutilisés |
| Cache historique | [processed.py](../src/clusterprep/processed.py) : `load_processed`, modèle d’écriture atomique et contrôle des checksums |
| Grille à résolution radio commune | [matched_dataset.py](../src/clusterprep/matched_dataset.py) : `MatchedDatasetAdapter`, autorise désormais Planck ; convolution radio inchangée |
| HEALPix et cache Planck | [planck.py](../src/clusterprep/planck.py) : `_read_map`, `project`, `aligned_images`, nouveaux |
| Normalisation, sélection des modalités | [modalities.py](../src/clusterprep/modalities.py) : `fit_normalization`, `normalize_planes`, `mode_keys`, étendus |
| Dataset et masques | [dataset.py](../src/clusterprep/dataset.py) : `_modality_images`, `_modality_item`, étendus ; signatures historiques préservées |
| Augmentations | [augmentation.py](../src/clusterprep/augmentation.py) : `transform_aligned`, réutilisé sans modification |
| Modèles | [models.py](../src/clusterprep/models.py) : `DualEncoderCNN` accepte Planck ; `MultiEncoderCNN` était déjà générique |
| Entraînement, checkpoints, folds | [training.py](../src/clusterprep/training.py) : `ExperimentConfig`, `run_experiment`, `_run`, étendus ; moteur de CV inchangé |
| CLI sur partitions existantes | [matched_runner.py](../src/clusterprep/matched_runner.py) : `load_plan`, `run_config_suite`, étendus pour les expériences Planck |
| Évaluation externe figée | [matched_evaluation.py](../src/clusterprep/matched_evaluation.py) : transmet Planck et vérifie les statistiques des modalités demandées |
| Transfer learning | [modular_transfer.py](../src/clusterprep/modular_transfer.py), inchangé : mappings explicites et rapport des modules chargés/incompatibles |
| Contrôle visuel | [planck_visualization.py](../src/clusterprep/planck_visualization.py), nouveau |
| Validation | [test_planck.py](../tests/test_planck.py), nouveau ; [planck_smoke.py](../examples/planck_smoke.py), nouveau |

Les fichiers de configuration et documents non suivis déjà présents dans le dépôt ont été conservés.

## API de configuration

Le projet utilise des clés plates dans `ExperimentConfig`, pas un objet `data`. CHANDRA porte le nom de modalité historique `xray`.

```json
{
  "name": "radio_planck",
  "mode": "multimodal",
  "model": "dual_encoder_cnn",
  "radio_source": "RAW",
  "modalities": ["radio", "planck"],
  "include_masks": true,
  "planck": {
    "source_path": "planck_PR4_data/PR4_NILC_y_map.fits",
    "cache_dir": "processed/planck_PR4",
    "field": "FULL",
    "force_rebuild": false
  },
  "normalization": {
    "planck": {"scope": "train", "method": "standard"}
  }
}
```

L’activation se fait uniquement par `modalities`. Sans Planck, les dispositions de canaux, poids de checkpoints, transformations et règles historiques sont conservés. Le nouvel argument du Dataset est ajouté à la fin pour préserver les appels positionnels existants.

Pour trois modalités : `modalities=["radio","xray","planck"]`, `model="multi_encoder_cnn"`. Les retours existants sont conservés : `raw`, `chandra`, leurs masques, `image`, `availability`, `label`. Planck ajoute `planck` et `planck_mask`. L’ordre d’`availability` suit `modalities`. Une CHANDRA présente sans recouvrement reste distinguée d’un fichier absent ; les masques spatiaux décrivent la couverture.

Pour les expériences qui activent Planck, `pair` exige les fichiers des modalités sélectionnées : Radio+Planck ne filtre donc pas sur CHANDRA. `multimodal` conserve les amas radio sans CHANDRA, même avec la politique historique `required`. Les exceptions ne sont pas converties en images nulles silencieuses. Une carte entièrement invalide porte une disponibilité Planck fausse en multimodal et provoque une erreur explicite en pair ; l’ajustement des statistiques échoue si aucun pixel d’entraînement n’est valide.

## Géométrie, cache et intensités

Le vrai fichier a été inspecté : HDU `Y-MAP`, champ `FULL`, NSIDE 2048, RING, GALACTIC, unité `y`, BEAMSIZE 10 arcmin. Les métadonnées sont lues et validées, et non imposées au lecteur. La conversion galactique vers WCS céleste utilise `reproject_from_healpix` et son moteur HEALPix ; ajouter `healpy` n’est pas nécessaire.

Le cache FITS conserve l’image physique et un HDU `VALID`, la WCS, l’unité et le faisceau fourni par la source. Son nom SHA-256 dépend du chemin de référence radio, de la forme, de la WCS sérialisée, du champ, de la version d’algorithme et de l’identité de la source (chemin absolu, taille, mtime et ctime nanoseconde). RAW et T50kpcSUB ne peuvent pas se confondre. Lecture : contrôle des checksums, dimensions, WCS, identité et cohérence masque/NaN ; reconstruction en cas de corruption. Écriture : fichier temporaire puis `os.replace`. `force_rebuild` force le calcul à chaque accès : le désactiver pour entraîner.

Avec `cache=true`, chaque Dataset Planck mémorise également les images physiques réduites et leurs masques avant normalisation, bruit et augmentation. La provenance et les métadonnées des fichiers de cache sont contrôlées à chaque accès ; un changement déclenche le rechargement et les vérifications complètes. `cache=false` conserve seulement les caches disque ; `force_rebuild=true` contourne aussi le cache mémoire. Aucun paramètre de configuration supplémentaire n’est nécessaire.

La carte complète est retenue en mémoire par processus ; elle n’est pas relue pour chaque amas. Les valeurs non finies, le sentinel HEALPix UNSEEN et `BAD_DATA` sont invalidés avant interpolation. Le fichier séparé `Masks.fits` n’est pas appliqué automatiquement : son choix scientifique et sa signification ne sont pas déduits du nom.

En historique, la carte entière est d’abord reprojetée sur la grille du FITS radio, puis utilise exactement `centered_raw_window` et le facteur de réduction radio. Le résultat physique réduit est maintenant conservé dans un cache FITS secondaire `reduced/` pour éviter de relire la grande grille entre processus/folds. En `matched_resolution`, le cache utilise directement la WCS commune produite par le prétraitement radio, sans charger une projection sur la grille radio native qui ne serait pas utilisée. Cette projection directe évite de limiter un champ commun plus large à l’empreinte originale. Aucune convolution Planck n’est appliquée.

Les images et leurs masques réutilisent le même index d’augmentation. Les masques utilisent l’interpolation nearest et le zoom pondéré par validité existants. Les caches ne contiennent aucune copie augmentée.

Normalisations Planck :

- `standard` par défaut : moyenne et écart-type globaux sur les pixels valides du train non augmenté ; pas de min-max par image. Écart-type nul : dénominateur 1.
- `none` : unités physiques conservées, aucun ajustement.
- `percentile_asinh` : `asinh((y-p_lo)/(p_hi-p_lo))`, bornes ajustées sur le train ; aucun clipping, valeurs signées conservées. Ce nom est distinct de l’ancienne normalisation radio `asinh_percentile`.

Les masques interviennent avant les statistiques ; les pixels invalides ne deviennent zéro qu’en sortie tensorielle. La normalisation est recalculée dans chaque fold uniquement sur ses clés train, puis sauvegardée dans le checkpoint et le rapport de normalisation. Validation et test utilisent ces statistiques figées.

Les encodeurs ont des poids indépendants. La fusion existante neutralise les vecteurs sans couverture et fournit la couverture au classifieur. Les anciens noms `radio_encoder` et `chandra_encoder` restent inchangés. Pour initialiser depuis un ancien modèle, utiliser les mappings `radio→radio` et éventuellement `xray→xray` du transfer learning modulaire ; l’encodeur Planck reste neuf. Aucun transfert n’est requis.

## Expériences prêtes

[Configurations Planck](../configs/planck/) : 40 JSON, soit 4 expériences × 5 folds × 2 prétraitements.

| Expérience | Radio | Autres modalités | Modèle |
|---|---|---|---|
| A | RAW | Planck | DualEncoderCNN |
| B | T50kpcSUB | Planck | DualEncoderCNN |
| C | RAW | CHANDRA + Planck | MultiEncoderCNN |
| D | T50kpcSUB | CHANDRA + Planck | MultiEncoderCNN |

Les configurations réutilisent exactement les cinq partitions du manifeste [phase7b_multimodal](../configs/phase7b_multimodal/), soit 154 amas, dont 89 sans CHANDRA. Fold 0 : 112 train / 29 validation ; autres folds : 113 / 28. Depuis le 17 novembre, A et C évaluent les 13 objets du test externe à la fin de chaque entraînement (`evaluate_test=true`). B et D les gardent réservés (`evaluate_test=false`). Le test ne participe ni à la normalisation ni à la sélection du checkpoint. Les hyperparamètres et métriques des configurations comparables sont conservés, notamment la patience présente de 50. Aucun entraînement complet n’a été lancé.

Depuis la racine du workspace, validation sans entraînement :

```sh
.venv/bin/python -m clusterprep clusterprep/configs/planck/historical/A/fold_0.json --dry-run
.venv/bin/python -m clusterprep clusterprep/configs/planck/matched_resolution/C/fold_0.json --dry-run
```

Retirer `--dry-run` démarre l’entraînement demandé dans le JSON ; ne le faire qu’après validation scientifique. Les chemins de données et de sortie sont relatifs à la racine du workspace, comme les expériences existantes.

Contrôle qualité :

```sh
.venv/bin/python -m clusterprep.planck_visualization --help
# Remplacer les chemins par les FITS de l'amas choisi.
.venv/bin/python -m clusterprep.planck_visualization \
  --radio RADIO.fits --chandra CHANDRA.fits --cluster NOM \
  --output outputs/planck_alignment.png
```

L’option `--matched-info` accepte un échantillon JSON avec `radio_paths` et `physical_record` pour afficher la grille matched. Les neuf panneaux montrent les trois images avant/après crop, les contours Planck sur la radio, puis le champ Planck large avec empreintes radio originale et finale. Les couleurs sont étirées uniquement pour l’affichage, sans modifier le cache.

## Validations effectuées et limites

Voir le [bilan de validation](PLANCK_VALIDATION.md) pour les résultats exacts et les commandes.

La projection sur de petits pixels n’améliore pas le faisceau d’environ 10 arcmin. Les figures vérifient la géométrie, pas l’absence de contamination astrophysique du signal y. Le lecteur accepte les cartes HEALPix complètes implicites RING/NESTED, galactiques ou équatoriales ICRS ; les autres conventions échouent explicitement.

La carte PR4 float64 nécessite environ 384 Mio résidents par processus, hors buffers temporaires et reprojection. Les configurations utilisent `num_workers=0` ; multiplier les workers multiplie potentiellement cette mémoire. Les images du cache radio complet peuvent aussi être volumineuses. L’identité de source repose sur les métadonnées du fichier plutôt que sur un hash relu intégralement à chaque accès.

Le mode matched conserve ses prérequis existants : les deux produits radio et les métadonnées physiques nécessaires. Les anciens notebooks d’analyse spécifiques à deux modalités n’ajoutent pas automatiquement un panneau Planck ; utiliser l’outil dédié. Les smoke tests sur six objets et images de 32 pixels ne constituent pas une mesure de performance ou un benchmark à taille nominale.

## A et C : test final à chaque entraînement (17 novembre 2026)

Les cinq folds de A et C, pour les deux prétraitements, activent désormais `evaluate_test=true`. Le moteur existant évalue le meilleur modèle sélectionné sur la validation après la fin de l’entraînement. Les résultats sont dans `metrics.json` (entrée `test`) et `test_predictions.csv` de chaque run. La normalisation reste ajustée uniquement sur le train. Les 13 mêmes objets de test sont utilisés pour chaque fold.

Le [script A/C](../run_planck_ac_5folds.sh) exécute dix entraînements séquentiels : A folds 0–4, puis C folds 0–4. Il s’arrête si un entraînement échoue. Depuis la racine du workspace :

```sh
# Vérifier seulement
bash clusterprep/run_planck_ac_5folds.sh matched_resolution --dry-run

# Entraîner et évaluer le test à la fin de chaque run
bash clusterprep/run_planck_ac_5folds.sh matched_resolution

# Variante historique
bash clusterprep/run_planck_ac_5folds.sh historical
```

Les scores test sont destinés au bilan final des expériences, pas au choix des hyperparamètres ou du checkpoint. Le script de smoke test conserve explicitement `evaluate_test=false` pour ne pas consommer le test externe pendant les validations techniques.
