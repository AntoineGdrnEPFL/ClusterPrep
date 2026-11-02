# Augmentation optionnelle du bruit radio

L'option JSON `augmentation.radio_noise` est portée par chaque expérience.
Absente ou `enabled: false`, elle conserve le chemin historique et ses résultats.
Voir les [configurations comparables](configs/radio_noise/) : trois méthodes,
chacune déclinée sur les cinq folds existants, même seed 42, mêmes splits,
modèles `fusion` et `dual_encoder_cnn`, tâches `rr_binary` et `rh_binary`,
hyperparamètres, seuils et protocole de test. Chaque JSON contient quatre expériences.

```json
{"augmentation":{"radio_noise":{
  "enabled":true,
  "method":"beam_correlated",
  "probability":0.5,
  "amplitude_range":[0.0,0.5],
  "rms_source":"estimated",
  "rms_estimation":{"method":"sigma_clipped_mad","region":"background","fallback":"raise"},
  "missing_beam_policy":"raise"
}}}
```

Méthodes exclusives : `gaussian_pixel` / `beam_correlated`. Probabilité dans
[0,1], amplitudes finies positives ou nulles et ordonnées. Tous les champs sont
validés au chargement. Aucune source catalogue fiable n'a été identifiée : les
valeurs autres que `estimated` sont refusées, plutôt que d'inventer un RMS.
Métadonnées beam absentes/invalides : `raise`, `skip` ou `gaussian_pixel` ; les
politiques de repli émettent un avertissement. Un fond inexploitable est une erreur.

## Ordre et compatibilité

FITS linéaire → crop centré → block averaging → cache physique sans bruit →
estimation RMS et bornes propres à l'image non augmentée → bruit radio →
normalisation → flip/rotation synchronisés radio/CHANDRA/masque.

Le projet utilise des percentiles **par image**, et non des statistiques ajustées
sur le train. Avec bruit, les percentiles 30–99 sont figés sur l'image propre à
l'entrée du Dataset, puis appliqués aux réalisations bruitées avant l'étirement
asinh historique. On n'ajuste jamais les bornes sur le bruit ajouté. Ce choix
préserve le protocole existant sans introduire une nouvelle normalisation globale.
CHANDRA, son masque, les labels et les partitions ne changent pas. Le train reçoit
`training=True` indépendamment de `augment` (qui contrôle les 24 variantes
spatiales) ; validation/test reçoivent `False`. Pour un usage direct du Dataset,
activer explicitement `training=True, radio_noise=...`.

Le scattering **précalculé**, dont `dual_ssn`, est refusé avec bruit actif : ses
coefficients ne correspondraient plus à l'entrée bruitée. Les modèles calculant
leurs caractéristiques à l'exécution gardent le même contrat. L'API de sortie du
Dataset est inchangée. Les anciens caches v2 restent lisibles et réutilisables.

## RMS, unités et faisceau

RMS = 1.4826022185 × MAD sur la bordure extérieure (quart de chaque axe), avec
jusqu'à huit rejets à 3 sigma et au moins 32 pixels restants. Cette région est un
**proxy géométrique de fond**, pas une segmentation astrophysique : vérifier le
masque du notebook lorsqu'une émission diffuse occupe aussi la bordure. Les
sources compactes brillantes sont rejetées. Zéro, NaN et fond insuffisant lèvent
une erreur. L'estimation se fait après réduction, avant clipping/asinh, dans les
mêmes unités que l'image ; aucune conversion implicite Jy/beam ↔ mJy/beam.

Le header du HDU radio fournit BMAJ/BMIN (FWHM en degrés), BPA (nord vers est),
BUNIT. La jacobienne locale au centre, calculée par différences WCS dans le plan
tangent est/nord, transforme la covariance céleste vers les pixels de la grille
réduite. Elle tient compte des axes inversés, rotations et tailles différentes.
L'approximation locale suppose une variation WCS faible sur le crop.

Un noyau gaussien tronqué à quatre sigma est convolué avec un bruit blanc sur
une grille élargie ; une convolution `valid` évite les bords artificiels. Le
champ corrélé est recentré et ramené à un écart-type unitaire sur les pixels
finis. Le blanc conserve des tirages gaussiens indépendants (RMS approché sur un
échantillon fini). Les NaN/Inf originaux sont conservés à l'injection ; la
normalisation historique les remplit ensuite de zéro.

Le block averaging conserve les unités de surface mais modifie le bruit et la
résolution. Le RMS est donc mesuré sur cette grille ; le noyau projeté ne simule
pas l'élargissement supplémentaire par la fenêtre de moyenne. Un faisceau plus
petit qu'un pixel donne un noyau quasi delta et un bruit presque blanc : le
notebook le signale. L'autocorrélation du bruit est celle du noyau (largeur plus
grande), pas exactement la covariance des résidus LOFAR. La PSF sale, la
pondération et les artefacts de calibration/déconvolution restent hors modèle.
Le RMS total attendu est approximativement `sqrt(RMS_original² + (alpha*RMS)²)`.

## Cache, reproductibilité et coût

Les nouveaux caches FITS conservent les métadonnées radio. Pour les caches
anciens, une première lecture du FITS source les récupère. Un petit JSON atomique
sous `processed/radio_noise/` mémorise RMS, covariance et métadonnées par identité
source (chemin, taille, timestamps), taille de grille et version de prétraitement.
Les images linéaires et les bornes sont conservées en mémoire par source/worker
lorsque le bruit est actif, même si `cache=false`, pour éviter les relectures.
Les noyaux sont mémorisés par covariance (LRU de 256 entrées). Aucune réalisation
aléatoire n'est persistée. Les diagnostics du notebook n'écrivent pas ces caches.

Tous les tirages utilisent NumPy, déjà initialisé par `run_experiment` et par
`_seed_worker` depuis le générateur PyTorch. Le bruit est renouvelé à chaque
accès, y compris aux variantes indexées. Reproductibilité pour même seed, fold,
ordre d'accès, nombre de workers et configuration ; changer le nombre de workers
ne garantit pas la même séquence. Un appel direct peut fournir un Generator
NumPy explicite à `add_radio_noise`. Sortie float32, calcul FFT en float64.

## Exécution et comparaison

Depuis la racine du projet :

```sh
bash clusterprep/run_radio_noise_5folds.sh
# Vérifier les commandes sans lancer l'entraînement :
bash clusterprep/run_radio_noise_5folds.sh --dry-run
```

Le script lance les cinq folds gaussiens puis les cinq folds corrélés : **40
entraînements** (deux modèles × deux classifications × cinq folds × deux bruits).
Les JSON binaires regroupent les mêmes expériences par fold. Voir les
[instructions de lancement](configs/radio_noise/README.md) pour ces deux modes
et la baseline facultative (20 entraînements supplémentaires).

Les JSON sont préparés ; ils ne constituent pas des résultats scientifiques.
Comparer les sorties `metrics.json` du même cohort test, séparément pour chaque
tâche et architecture : `test.macro_f1`, puis `f1`, `precision`, `recall` et
`average_precision` dans `test.per_class` (`RR` ou `RH` et leur classe négative).
AP est la convention PR-AUC du projet ; aucune métrique n'est redéfinie. Rapporter
moyenne et écart-type entre cinq folds et les deltas appariés bruit–baseline par
fold. Ne pas mélanger des relances, seeds ou cohortes. Répéter les trois bras
avec les mêmes seeds supplémentaires pour quantifier l'incertitude.

La section autonome du [notebook existant](notebooks/analyse_cluster_radio_chandra.ipynb)
permet la sélection d'amas, méthodes, alpha, seed et réalisations. Elle utilise
les mêmes fonctions que le Dataset, montre les images et différences à échelles
communes, l'ellipse du beam, l'autocorrélation, le fond RMS, CHANDRA reprojeté et
le mode normalisé. Elle a été exécutée sur un FITS réel ; les mesures et résultats
de validation sont consignés dans [le rapport](RADIO_NOISE_VALIDATION.md).


## Vérifier une corrélation peu visible

Dans le notebook, `NOISE_ACF_RADIUS=20` règle le zoom (15 est aussi possible) et
`NOISE_ACF_LINTHRESH=0.02` la sensibilité de l'échelle symlog. Les coupes centrales
comparent la mesure au noyau discret de la pipeline, et les tableaux donnent les
FWHM en pixels, la vérification WCS indépendante, les corrélations à un pixel et
la sensibilité du fond aux buffers autour des rejets positifs.

`NOISE_DIAGNOSTIC_ROTATION=30` permet de suivre la perturbation après la
normalisation et la géométrie. Le recentrage/mise à l'échelle conservent Pearson,
mais clipping/asinh et rééchantillonnage NEAREST peuvent modifier la structure
spatiale. Un beam sous-pixel ne crée quasiment aucune corrélation résolue.
Voir [les résultats détaillés](RADIO_NOISE_VALIDATION.md) avant d'interpréter une
faible autocorrélation comme une erreur de génération ou comme une validation
physique du modèle.
