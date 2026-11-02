# Validation — bruit radio (31 octobre 2026)

**Mise à jour des expériences :** les anciens JSON multilabel RR/RH sont remplacés
par les deux classifications binaires `rr_binary`/`rh_binary`, pour `fusion` et
`dual_encoder_cnn`, avec les paramètres de la suite binaire (dont patience 50).
Chaque JSON de `radio_noise` contient quatre expériences ; chaque JSON binaire
regroupe les huit variantes gaussiennes/corrélées. Le nouveau script lance les
40 entraînements avec bruit, gaussiens puis corrélés. La baseline facultative en
contient 20. Les mentions des quinze anciens entraînements ci-dessous décrivent
l'état initial ; aucun de ces entraînements scientifiques n'a été lancé ici.

## Modifications

- [radio_noise.py](src/clusterprep/radio_noise.py) : validation JSON, RMS robuste,
  covariance WCS/beam, noyau mémorisé, bruits et cache de contexte atomique.
- [preprocessing.py](src/clusterprep/preprocessing.py) et
  [processed.py](src/clusterprep/processed.py) : métadonnées du HDU radio conservées,
  bornes optionnelles de normalisation, compatibilité des caches v2.
- [dataset.py](src/clusterprep/dataset.py) et [training.py](src/clusterprep/training.py) :
  injection exclusivement train avant normalisation et géométrie ; sorties inchangées.
- [notebook existant](notebooks/analyse_cluster_radio_chandra.ipynb) : trois cellules
  ajoutées ; les quinze cellules préexistantes sont identiques à HEAD.
- [configurations](configs/radio_noise/) : baseline, gaussian_pixel, beam_correlated,
  chacune sur cinq folds, soit quinze JSON validés et comparés champ à champ.
- [tests](tests/test_radio_noise.py), [documentation](RADIO_NOISE.md), lien README,
  déclaration explicite de SciPy dans les dépendances.

## Tests et vérifications

Commande ciblée :

```sh
.venv/bin/python -m pytest clusterprep/tests/test_radio_noise.py \
  clusterprep/tests/test_pipeline.py clusterprep/tests/test_processed.py \
  clusterprep/tests/test_training.py -q
```

Premier passage final d'intégration : **47 tests réussis, 2 sous-tests réussis**,
17,23 s. Deux vérifications supplémentaires couvrent ensuite la relecture unique
des métadonnées d'un ancien cache et l'isotropie du bruit circulaire. Le dernier
passage du fichier dédié donne **28 tests réussis en 12,50 s**.

Couverture : configuration invalide, absence/désactivation, RMS robuste avec
sources brillantes et changement d'unité, WCS tourné/inversé, facteur de réduction,
orientation et dimensions du faisceau, RMS et corrélations des deux méthodes,
NaN, immutabilité, politiques beam manquant, conservation CHANDRA/masques/labels,
ordre bruit-normalisation-géométrie, absence de bruit en évaluation, cache images,
contexte et noyau, reproductibilité à 0 et 2 workers, configurations comparables.
Deux entraînements synthétiques d'une époque, un par méthode, vérifient que
seul le Dataset train reçoit le bruit, avec `augment=False` également.

L'environnement macOS échoue avec `torch_shm_manager` lors de la collation tensor
partagée entre processus. Le test de reproductibilité utilise une collation NumPy
pour exercer les vrais workers et leur initialisation sans cette dépendance IPC ;
il ne prétend pas valider le transport partagé des tenseurs dans cet environnement.
Les entraînements synthétiques utilisent zéro worker.

La suite complète (avant les deux derniers tests supplémentaires) donne **358
tests réussis, 3 échecs, 1 ignoré et 33 sous-tests réussis**, en 80,46 s.
Les échecs hors bruit radio sont liés à des fichiers absents :
les tests des configurations binaires cherchent `configs/rr_binary.json` et
`configs/rh_binary.json`, alors que les exemples sont sous `configs/rr_rh_binary/` ;
le smoke test de l'autre notebook multilabel exige un cache FITS `synthetic-36`
absent. Ces tests et chemins n'ont pas été modifiés pour masquer les échecs.

Structure validée avec nbformat ; `git diff --check` sans erreur.

## Diagnostic sur les FITS réels

Exécution autonome des deux nouvelles cellules Python, sans entraînement et sans
écriture dans les caches/données. Figures rendues avec Matplotlib Agg ; grille
inspectée visuellement. Taille 128, seed 42, alpha 0,5, une réalisation par méthode.
Toutes les valeurs ci-dessous sont en JY/BEAM sur la grille réduite.

| Amas | RMS radio | Cible ajoutée | Blanc, RMS fond | Corrélé, RMS fond |
|---|---:|---:|---:|---:|
| PSZ2G081.02+50.57 | 1,812963e-5 | 9,064815e-6 | 9,097484e-6 | 9,064427e-6 |
| PSZ2G084.13-35.41 | 9,647823e-5 | 4,823912e-5 | 4,841723e-5 | 4,824981e-5 |
| PSZ2G023.17+86.71 | 3,080357e-5 | 1,540179e-5 | 1,550392e-5 | 1,540500e-5 |
| PSZ2G045.87+57.70 | 3,431441e-5 | 1,715721e-5 | 1,726342e-5 | 1,713500e-5 |

Les écarts sur le fond sont inférieurs à 1 %. Le corrélé est normalisé exactement
sur la zone valide, pas sur le sous-ensemble de fond : une petite différence est
attendue. Le RMS résultant est affiché séparément dans le notebook.

Cas sous-échantillonnés effectivement observés : beam réduit 0,220 × 0,328 pixel
pour PSZ2G081.02+50.57 (réduction ×16), contre 0,897 × 1,464 pixel pour
PSZ2G084.13-35.41 (×4). Le notebook avertit explicitement ; l'anisotropie est bien
plus visible dans le second cas. Aucun élargissement artificiel du beam n'a été
introduit pour rendre les figures plus convaincantes.

Microbenchmark local, 200 tirages 128×128, génération seule avec un faisceau
résolu de covariance [[9,2],[2,3]] : **0,194 ms/blanc**, **0,501 ms/corrélé**.
Ces chiffres excluent FITS, reprojection, première estimation RMS et entraînement.

## Écarts au plan et reste expérimental

Le pipeline ne possède pas de statistiques globales de normalisation apprises
sur train : les bornes sont celles de chaque image propre, conformément à son
architecture historique. Le scattering précalculé est explicitement incompatible
avec l'option active. Aucun RMS externe fiable n'est supposé. Le fond géométrique
et la résolution après block averaging sont des approximations documentées.

Les quinze entraînements scientifiques RR/RH (jusqu'à 200 epochs chacun) ne sont
pas exécutés. Il n'y a donc **pas encore de comparaison F1/PR-AUC empirique** ni
de conclusion sur le gain de généralisation. Les commandes, configurations
appariées et champs de métriques à comparer figurent dans [RADIO_NOISE.md](RADIO_NOISE.md).

## Correction du diagnostic « Métadonnées : {} »

Le message a été reproduit avec les FITS réels en simulant un ancien
`preprocess` encore chargé dans un noyau Jupyter. Les headers sources contiennent
bien BMAJ/BMIN/BPA/BUNIT, mais l'ancien résultat ne transmet pas `radio_metadata`.
Les appels du notebook passent maintenant explicitement l'entrée source à
`prepare_noise_context`, sans activer le cache disque. Les métadonnées absentes,
vides ou partielles sont complétées depuis le HDU radio source ; le WCS de la
grille réduite reste celui du prétraitement. La version du cache de contexte
passe à 2 pour reconstruire les anciens contextes potentiellement incomplets.
Un beam réellement absent du FITS continue de déclencher la politique explicite.

Validation de la correction : **39 tests réussis** (bruit et cache prétraité),
et exécution des deux méthodes sur les quatre amas réels en simulant l'ancien
prétraitement : **8 réalisations réussies**. Redémarrer le noyau Jupyter après la
mise à jour, puis relancer les deux cellules de la section autonome de diagnostic.

## Vérifications détaillées de l'autocorrélation et du fond

Le [notebook](notebooks/analyse_cluster_radio_chandra.ipynb) propose maintenant une
fenêtre ACF **±20 pixels** (réglable à 15), une échelle symlog (zone linéaire
±0,02), des coupes centrales ±5 pixels et les valeurs de Pearson à un pixel en x/y.
Les courbes attendues utilisent le noyau discret effectivement employé par
`add_radio_noise`, sans approximation gaussienne continue aux petites FWHM.

Le nouveau [module de diagnostics](src/clusterprep/radio_noise_diagnostics.py)
compare aussi la covariance calculée par la jacobienne à une construction
indépendante : projection WCS des extrémités des axes célestes BMAJ/BMIN/BPA.
Sur les quatre amas, l'écart relatif est inférieur à 1,1e-9. Des tests couvrent
également les pixels anisotropes, les axes inversés, le cisaillement WCS et une
position proche du passage RA=0.

Mesures sur 128×128, alpha=0,5, seed=42, avant clipping/asinh :

| Amas | FWHM majeur × mineur (px) | ρx blanc | ρy blanc | ρx corrélé | ρy corrélé | ρx/ρy attendus du noyau |
|---|---:|---:|---:|---:|---:|---:|
| PSZ2G081.02+50.57 | 0.328 × 0.220 | 0.0115 | 0.0020 | 0.0127 | -0.0133 | 0.0000 / 0.0000 |
| PSZ2G084.13-35.41 | 1.464 × 0.897 | 0.0115 | 0.0020 | 0.4348 | 0.0817 | 0.4285 / 0.0783 |
| PSZ2G023.17+86.71 | 0.385 × 0.206 | 0.0115 | 0.0020 | 0.0127 | -0.0133 | 0.0000 / 0.0000 |
| PSZ2G045.87+57.70 | 0.364 × 0.215 | 0.0115 | 0.0020 | 0.0127 | -0.0133 | 0.0000 / 0.0000 |

Le tirage blanc est le même champ standardisé d'un amas à l'autre (seed et grille
identiques) : ces lignes ne sont pas des réalisations statistiquement indépendantes.
Les petits écarts au zéro attendu ne démontrent pas une corrélation physique.
Sur PSZ2G084.13-35.41, l'anisotropie mesurée est conforme au noyau discret. Sur
les trois autres amas, le noyau est presque une impulsion : il ne produit pas une
corrélation résolue significative. Ceci valide l'implémentation discrète, pas une
reproduction exacte de la covariance du bruit interférométrique LOFAR.

### Ordre des opérations et effet des transformations ultérieures

Un test reproduit le champ blanc avec la même RNG et vérifie numériquement :
convolution avec padding → recentrage → division par l'écart-type → multiplication
par alpha × RMS. Les corrélations de Pearson avant/après remise à l'échelle
concordent à 1e-8. Les tests comparent le champ obtenu et l'autocorrélation attendue
pour un beam résolu anisotrope, circulaire et sous-pixel.

La corrélation n'est **pas conservée à l'identique après toute la pipeline**.
Pour PSZ2G084.13-35.41, le résidu corrélé passe de (ρx,ρy)=(0,435;0,082) à
(0,318;0,063) après clipping/asinh, puis (0,205;0,123) après rotation NEAREST
à 30°. Cette dernière modifie à la fois l'orientation et l'échantillonnage ; les
coefficients x/y ne sont donc pas censés rester constants. Pour le blanc, les
valeurs passent d'environ (0,012;0,002) à (0,076;0,075) après normalisation et
rotation : la répétition de pixels par NEAREST introduit elle-même une corrélation.
Aucune de ces observations ne justifie d'assimiler le résidu normalisé à un bruit
gaussien stationnaire indépendant du signal. Le comportement historique du
Dataset n'a pas été modifié pour ces diagnostics.

L'équivalence du diagnostic et du Dataset est testée exactement sur les pixels
valides pour les variantes 0°, 30°, 30° avec flip, et 90°.

### Sources compactes et sensibilité du RMS

La figure distingue les pixels hors région, retenus et rejetés. L'histogramme
montre le rejet de la queue positive. Le RMS est recalculé **pour diagnostic**
avec des buffers de 1 et 2 pixels autour des rejets positifs ; le RMS d'entraînement
reste inchangé. Un rejet positif peut aussi être une fluctuation du bruit, pas
nécessairement une source astrophysique.

| Amas | Rejets positifs | Pixels de fond retenus | Variation RMS avec buffer 2 px |
|---|---:|---:|---:|
| PSZ2G081.02+50.57 | 230 | 12016 / 12288 | -1.89 % |
| PSZ2G084.13-35.41 | 293 | 11928 / 12288 | -3.11 % |
| PSZ2G023.17+86.71 | 198 | 11980 / 12288 | -2.13 % |
| PSZ2G045.87+57.70 | 156 | 12103 / 12288 | -1.16 % |

La sensibilité est faible mais non nulle : aucune preuve d'un fond entièrement
sans émission. Les ailes faibles et l'émission diffuse peuvent rester dans la
sélection. Un test d'injection de quatre sources gaussiennes compactes d'amplitude
80 sigma et sigma=2 pixels, sur un fond connu, confirme le rejet de plus de 98 %
des pixels des sources dépassant 5 sigma et un RMS à moins de 4 % de la vérité.
Cela ne valide pas tous les niveaux de contamination possibles des images réelles.

Validation : **41 tests réussis** (bruit et nouveaux diagnostics), notebook valide
et exécuté sur les quatre amas, figures rendues et inspectées visuellement.

## Validation des configurations binaires et du lanceur

Les 60 définitions d'expérience de `radio_noise` (20 baseline, 20 gaussiennes,
20 corrélées) ont été comparées aux quatre expériences originales de chaque fold :
aucun changement des modèles, tâches, splits ou hyperparamètres hors nom et bruit.
Les JSON binaires regroupent exactement les huit variantes correspondantes par
fold. Le test automatisé d'appariement des configurations passe.

La syntaxe des deux scripts Bash est valide. Le nouveau lanceur a été exercé avec
un interpréteur factice : ordre exact des dix commandes, exécution depuis un autre
dossier, sélection des folds, mode dry-run sans exécution, rejet des doublons et
arguments invalides, arrêt dès la première erreur avec conservation du code de
sortie. Aucun entraînement réel n'a été lancé pendant ces vérifications.
