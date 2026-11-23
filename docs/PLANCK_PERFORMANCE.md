# Diagnostic de performance Planck — 20 novembre 2026

Analyse en lecture du code et profilage local pendant l’entraînement DE/NDE fold 0. Aucun code de la pipeline ni entraînement en cours n’a été modifié ou interrompu. Les appels de reprojection et d’écriture du cache ont été interdits dans le processus de profilage, pour distinguer réutilisation et reconstruction.

## Conclusion

Le surcoût récurrent principal identifié est le chargement/prétraitement des données. Le cache Planck évite la reprojection, mais son image pleine résolution est relue, vérifiée et réduite à chaque échantillon. De plus, `cache=true` ne mémorise pas les images dans le chemin à modalités explicites. Cette inefficacité de l’intégration se retrouve au train et en validation.

## Chemin critique

1. [ClusterDataset.__getitem__](../src/clusterprep/dataset.py) retourne immédiatement `_modality_item` pour les modalités explicites et ne passe pas par le cache mémoire historique.
2. `_modality_item` appelle `_modality_images` à chaque accès. Son cache mémoire reste vide même après une première passe entière.
3. [aligned_images](../src/clusterprep/planck.py) recharge le FITS radio original pour retrouver sa géométrie. [load_celestial_fits](../src/clusterprep/io.py) charge et copie les pixels, bien que seule la forme/WCS soit nécessaire ici.
4. `project` ouvre le cache Planck pleine grille : lecture image/masque, conversion des tableaux, checksum FITS, contrôle de finitude et WCS, vérifications supplémentaires explicites.
5. `centered_raw_window` et `block_average` refont la réduction vers 128 × 128 à chaque accès.
6. Normalisation et géométrie suivent. Les 24 variantes du train déclenchent chacune la même préparation déterministe ; la validation la recommence à chaque époque.
7. `num_workers=0` : préparation CPU dans le processus principal, sans workers pour recouvrir ce coût avec le calcul du modèle MPS.

Le cache CHANDRA historique contient déjà les images réduites à la taille du réseau. Le cache Planck ne contient que l’image reprojetée sur la grande grille radio, ce qui explique le surcoût même lorsque les deux modalités utilisent le même Dataset explicite.

## Mesures

Configuration observée : historique, RAW+Planck, taille 128, batch 16, augmentations activées, cache true, num_workers 0. Split réel : 123 train, 42 validation, 42 test. Le journal rapporte 2944 exemples effectivement entraînés par époque après drop_last (123 × 24 = 2952 avant drop_last).

Microbenchmark sans modèle et sans augmentation, huit accès répétés par objet après amorçage, médiane :

| Accès Dataset | Carte 1344 × 1344 | Carte 1730 × 1730 |
|---|---:|---:|
| RAW historique en mémoire | 0,031 ms | 0,033 ms |
| RAW explicite | 2,347 ms | 2,629 ms |
| RAW + Planck explicite | 13,703 ms | 17,887 ms |

Les vingt appels Planck instrumentés ont tous retourné `reused`. Aucune reprojection HEALPix n’est intervenue. Le test RAW+CHANDRA sur l’objet réellement apparié a été abandonné car son cache prétraité n’était pas réutilisable ; aucun chiffre de comparaison CHANDRA réel n’est donc revendiqué.

Passe complémentaire sur les ensembles complets, une image non augmentée par amas :

| Ensemble | Nombre | Temps cumulé Dataset | Médiane | P95 |
|---|---:|---:|---:|---:|
| Validation | 42 | 0,922 s | 19,01 ms | 36,47 ms |
| Train | 123 | 5,063 s | 27,88 ms | 97,42 ms |

Aucun cache manquant/invalide pendant ces passes et zéro entrée dans le cache mémoire du Dataset à la fin. Conserver localement les tableaux déjà préparés d’un amas, uniquement dans le processus de diagnostic, ramène son assemblage/normalisation à 0,187 ms médian. Ce dernier chiffre est une borne indicative pour la partie données, pas une mesure de gain d’entraînement après correction.

Le profil cProfile de deux objets attribue environ 17–20 ms sur 23–26 ms instrumentées à `aligned_images`. Ouverture FITS, reconstruction/sérialisation WCS, copie des données et checksums sont visibles dans le chemin critique. Le profilage lui-même ajoute du coût ; ces temps ne remplacent pas les médianes non instrumentées.

Nombre de paramètres des modèles construits avec les options de ce run :

| Modalités | Paramètres |
|---|---:|
| RAW, DualEncoderCNN | 84 658 |
| RAW + CHANDRA, DualEncoderCNN | 169 258 |
| RAW + Planck, DualEncoderCNN | 169 258 |
| RAW + CHANDRA + Planck, MultiEncoderCNN | 253 794 |

RAW+Planck a donc exactement la même architecture et le même nombre de paramètres que RAW+CHANDRA dans cette comparaison. Le troisième encodeur ajoute naturellement du calcul pour la combinaison triple, mais n’explique pas les relectures répétées ni le surcoût de la combinaison double.

Les deux premières durées du journal observées étaient 137,31 s puis 88,87 s. Ce sont des temps train+validation, pas des chronométrages séparés. L’amorçage/compilation peut contribuer à la première époque, sans expliquer la préparation répétée des suivantes.

## Corrections recommandées, par priorité

1. **Cache mémoire des tableaux déterministes réduits**, par amas et par Dataset, avant augmentation et bruit aléatoire. Inclure provenance, géométrie, source Planck, sélection radio, version et options dans l’invalidation. Ne pas partager de statistiques apprises entre folds. Ne pas mémoriser les augmentations ni un tirage de bruit.
2. **Cache persistant secondaire Planck après crop/réduction**, physique et non normalisé, à côté du cache pleine grille réutilisable. Il évite de refaire la réduction lors d’un nouveau processus/fold.
3. **Lecture de la géométrie radio sans copie des pixels** lorsque seules forme et WCS sont requises. Conserver le choix correct du HDU céleste et les éventuelles dimensions supplémentaires.
4. **Validation intégrale du cache à son chargement en mémoire**, puis contrôle léger de provenance lors des réutilisations. Garder la détection de corruption et la reconstruction ; ne pas simplement désactiver les checksums.
5. Dans le chemin `matched_resolution`, déplacer la branche matched avant la projection pleine grille : l’image `full` y est actuellement calculée/chargée puis ignorée. Ce défaut ne concerne pas l’entraînement historique observé.
6. Après ces corrections, mesurer si plusieurs workers sont encore utiles. Augmenter `num_workers` seul masque partiellement le problème, répète le travail et peut multiplier la mémoire de la carte HEALPix lors des créations de cache.

Ordre de grandeur mémoire : pour 207 amas, stocker un tenseur float32 de 3 × 128 × 128 représente environ 39 Mio ; 5 canaux environ 65 Mio, hors objets Python et métadonnées. Il n’est pas nécessaire de conserver en mémoire toutes les cartes radio pleine résolution.

## Limites

Mesures réalisées pendant qu’un entraînement utilisait la machine : CPU, caches du système et contention varient. Les passes sur la population complète ne doivent pas être extrapolées linéairement en une promesse de durée par époque. Pas de benchmark GPU ni de mesure train/validation séparée, pas de correction de production appliquée dans cet audit.


## Optimisations appliquées après cet audit

La demande suivante a autorisé les corrections. Les mesures ci-dessus restent le diagnostic **avant** optimisation.

- `ClusterDataset._modality_images` mémorise les tableaux physiques réduits pour les expériences avec Planck et `cache=true`, séparément pour chaque Dataset. Les signatures incluent sources radio/Planck/CHANDRA/optique, taille, modalités, configuration Planck, options et enregistrements matched. Les fichiers des caches réduits et radio historiques sont surveillés par leurs métadonnées. Les résultats retournés sont copiés pour empêcher qu’un consommateur modifie les tableaux mémorisés.
- Nouveau cache FITS `processed/planck_PR4/reduced/` : image physique, masque, WCS, unité et beam conservés, checksums, validation et écriture atomique. Identité dépendante des sources, WCS finale, taille, facteur de réduction et version. Les images restent non normalisées et non augmentées.
- Nouveau lecteur de géométrie radio sans charger les pixels, utilisé uniquement lorsqu’un crop doit être construit. Les accès au crop déjà présent ne relisent pas le FITS radio.
- Le chemin matched projette seulement sur la grille commune, sans calculer/charger l’image pleine grille inutilisée.
- `force_rebuild=true` contourne les caches mémoire et disque. `cache=false` garde le cache disque secondaire. Pas de changement des statistiques de normalisation, des augmentations, du bruit, des splits ni des modèles.

Mesures après optimisation, sur les mêmes deux amas, sans modèle :

| Amas | Ancien accès Dataset | Nouveau, cache réduit disque | Nouveau, cache mémoire amorcé |
|---|---:|---:|---:|
| PSZ2G123.00-35.52 | 13,703 ms | 5,725 ms | 0,298 ms |
| PSZ2G071.63+29.78 | 17,887 ms | 5,231 ms | 0,294 ms |

Le cache mémoire donne ici un gain d’environ 46–61× sur **l’accès aux données seul**. Les images physiques de ces deux amas sont strictement égales à la reconstruction par l’ancien calcul de bloc sur la projection pleine grille.

Après amorçage, passe sans augmentation sur les 42 objets validation : **0,0179 s** (ancien diagnostic : 0,922 s). Passe sur les 123 objets train : **0,0489 s** (ancien diagnostic : 5,063 s). Les mémoires contiennent respectivement 42 et 123 entrées. Ces mesures ont été faites à des moments différents, sans entraînement actif pour le benchmark final ; elles ne constituent pas une promesse équivalente sur la durée totale d’une époque.

La première lecture de chaque objet dans un Dataset remplit encore la mémoire. Si le nouveau cache réduit manque, il est construit une fois depuis le cache plein existant. Les anciens caches restent utilisables ; aucune suppression n’est nécessaire. Les configurations actuelles activent déjà `cache=true`.

Tests spécifiques supplémentaires : équivalence exacte des 72 variantes rotation/flip/zoom, normalisation modifiable sans figer les statistiques, bruit recalculé à chaque accès, copie défensive, invalidation des sources et caches corrompus, option force, option sans cache mémoire, WCS/tailles, lecture d’en-têtes 2D/4D et chemin matched sans projection inutile. Aucun entraînement réel relancé.

Validation de régression : **197 tests réussis** sur les suites Planck, cache Planck, legacy, matched dataset, sémantique des modes, processed, pipeline, bruit radio et zoom. `git diff --check` ne signale aucune erreur.
