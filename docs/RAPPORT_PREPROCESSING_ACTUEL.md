# Rapport du preprocessing actuel — neuf combinaisons de modalités

État du code local examiné le 20 novembre 2026. Ce rapport décrit `clusterprep`, y compris les modifications locales présentes au moment de la lecture. Il repose sur le code exécuté et les configurations, sans nouvel entraînement ni recalcul des images.

## 1. Périmètre : « par défaut » a deux significations

Le **défaut logiciel** correspond aux paramètres appliqués lorsqu'une option n'est pas spécifiée. Les **configurations d'expériences du dépôt** peuvent les remplacer. Cette distinction est essentielle pour la normalisation.

Le chemin spatial par défaut est `radio_preprocessing="historical"`. Sélectionner `radio_source="T50kpcSUB"` ne bascule pas automatiquement vers `matched_resolution`. Ajouter Planck ne le fait pas non plus.

| Paramètre | Défaut logiciel |
|---|---|
| Source radio | `RAW` |
| Prétraitement spatial | `historical` |
| Taille finale | 128 × 128 pixels |
| Mode d'expérience | `raw` ; les expériences multimodales examinées choisissent `multimodal` |
| Modalités explicites | Aucune liste par défaut ; LEGACY et Planck nécessitent une sélection explicite |
| Produit radio manquant | Erreur, sans remplacement silencieux par RAW |
| Modalité auxiliaire manquante | Politique `optional`, sous réserve des contraintes du mode |
| Masques auxiliaires | Activés |
| Bandes LEGACY | g, r, z |
| Radio, CHANDRA, LEGACY | Percentiles 30–99, étirement asinh, α = 10 ; statistiques par image/bande |
| Planck, lorsqu'il est sélectionné | Standardisation avec moyenne et écart-type du train |
| Augmentation dans l'entraînement | Activée ; validation et test non augmentés |
| Zoom / ajout de bruit radio | Désactivés |
| Cache physique | Activé via le répertoire de produits prétraités |

Le constructeur de dataset isolé a `augment=False`, mais la configuration d'expérience a `augment=True` et ne l'active que sur le train. Le tableau décrit donc la pipeline d'entraînement.

Sources : [configuration d'expérience](../src/clusterprep/training.py#L59), [dataset](../src/clusterprep/dataset.py), [modalités](../src/clusterprep/modalities.py), [défauts Planck](../src/clusterprep/planck.py#L199).

## 2. Sélection des données et lecture des FITS

### 2.1 Identité des amas et produit radio

L'inventaire part des amas et annotations RAW. Les produits radio alternatifs sont associés aux identifiants d'amas. Le choix de T50kpcSUB change le fichier radio actif, sans remplacer l'identité ni les labels de l'amas.

**T50kpcSUB est un produit d'entrée déjà construit.** Le preprocessing étudié ne fabrique pas ce produit depuis RAW et ne réalise pas de nouvelle soustraction des sources compactes ou de taper à 50 kpc. Ces opérations éventuelles appartiennent à la production amont du FITS et ne peuvent pas être déduites intégralement de ce chemin de lecture.

Par défaut, demander un produit absent provoque une erreur. Une exclusion peut être demandée explicitement avec la politique appropriée, mais il n'y a pas de repli automatique vers RAW.

### 2.2 Lecture radio et CHANDRA

1. Ouverture du FITS.
2. Recherche du premier HDU contenant une image et un WCS céleste valide.
3. Suppression des axes de longueur 1.
4. Si des axes supplémentaires subsistent, sélection répétée du premier plan, avec avertissement : ce n'est pas une intégration spectrale.
5. Obtention d'une image 2D et de son WCS céleste.
6. Conversion en flottants ; les images spatiales retournées sont en `float32`.

### 2.3 Lecture LEGACY

Le fichier est fourni par l'inventaire ou recherché via la racine LEGACY configurée, dans un sous-répertoire par amas. Le motif de nom par défaut est `{cluster_id}_legacy.fits`. Il n'y a pas de téléchargement automatique dans ce preprocessing.

Le lecteur conserve l'axe des bandes : forme `[bandes, hauteur, largeur]`. Il sélectionne g, r, z dans l'ordre demandé, à partir des métadonnées `BANDi`, ou utilise des indices explicites. Un fichier 2D est interprété comme une seule bande ; cela ne crée pas artificiellement les trois bandes attendues. Une sélection de bandes incompatible échoue.

Sources : [lecture FITS et sélection radio](../src/clusterprep/io.py), [chargement des modalités](../src/clusterprep/dataset.py).

## 3. Géométrie radio commune au chemin historique

La **carte radio sélectionnée** fixe la grille, le champ et le facteur de réduction de toutes les modalités de l'amas.

Pour une carte de hauteur H, largeur W, et une taille de sortie S = 128 :

```text
r = floor(min(H, W) / S)
f = plus grande puissance de 2 inférieure ou égale à r
côté de la fenêtre native = S × f
x0 = floor((floor(W/f) − S) / 2) × f
y0 = floor((floor(H/f) − S) / 2) × f
```

1. Si `r < 1`, erreur : la carte est trop petite. Aucun padding ni agrandissement implicite.
2. Extraction d'une fenêtre carrée approximativement centrée, alignée sur des blocs de f pixels.
3. Réduction par **moyenne des valeurs finies dans chaque bloc f × f**.
4. Si un bloc ne contient aucune valeur finie, sa sortie reste NaN.
5. Mise à jour du WCS par découpage et pas f.
6. Conservation des métadonnées radio disponibles : BMAJ, BMIN, BPA, BUNIT.

Exemples : une image 1024 × 1024 donne f = 8 ; une image 768 × 768 donne f = 4 et un crop central 512 × 512 ; une image 200 × 200 donne f = 1 et un crop 128 × 128.

C'est une moyenne, **pas une somme de flux**. La réduction est un filtrage par blocs ; il n'y a pas en plus de convolution gaussienne de mise à résolution commune dans ce chemin.

Le champ n'est pas fixé automatiquement en kpc ou en unités de R500. Le centrage est celui du tableau radio, pas un recentrage calculé sur le pic radio/X ou le barycentre optique.

Conséquence : si RAW et T50kpcSUB ont des dimensions, un WCS ou un échantillonnage différents, leurs sorties 128 × 128 peuvent couvrir des champs différents. Une même taille de tenseur ne garantit ni le même champ céleste ni la même PSF.

Source : [prétraitement spatial](../src/clusterprep/preprocessing.py#L6).

## 4. Traitement spécifique de chaque modalité auxiliaire

### 4.1 CHANDRA

1. Lecture de l'image 2D et de son WCS.
2. Reprojection **bilinéaire sur la fenêtre radio à résolution native**, avant réduction.
3. Mise à NaN des pixels dont le footprint de reprojection est nul ou négatif.
4. Moyenne par les mêmes blocs f × f que la radio.
5. Calcul du masque final : `isfinite(chandra_reduite)`.
6. Normalisation uniquement sur les pixels finis et couverts.
7. Remplacement des pixels invalides par zéro dans le tenseur normalisé, tout en conservant leur masque à zéro.

Le masque est spatial et binaire, à la résolution finale. Il ne mesure ni l'exposition, ni une probabilité de qualité, ni la fraction de couverture du bloc. Un seul pixel fini dans un bloc peut suffire à obtenir une moyenne finie, donc un masque final égal à 1.

Une image CHANDRA existante mais sans recouvrement est conservée : elle devient une image nulle avec un masque nul. Aucune correction d'exposition, soustraction de fond X, suppression de sources ponctuelles ou conversion des comptes en flux n'est ajoutée par ces fonctions ; elles utilisent le produit fourni.

Source : [branche CHANDRA](../src/clusterprep/preprocessing.py#L38).

### 4.2 LEGACY

Pour chacune des bandes g, r, z, indépendamment :

1. Conversion des valeurs non finies en NaN.
2. Reprojection bilinéaire sur exactement la même fenêtre radio native.
3. Mise à NaN hors footprint.
4. Moyenne par blocs f × f.
5. Masque de validité final propre à la bande.
6. Normalisation de la bande sur ses seuls pixels valides.

La sortie physique possède trois plans d'intensité et trois masques. Il n'y a pas de composition RGB d'affichage, de mélange des couleurs, de normalisation commune aux trois bandes, de correction d'extinction ou de calibration photométrique supplémentaire dans ces fonctions.

Avec T50kpcSUB, les trois bandes sont reprojetées sur la grille T50kpcSUB, et non automatiquement sur celle de RAW.

Source : [prétraitement optique](../src/clusterprep/preprocessing.py#L110).

### 4.3 Planck

Les configurations Planck DE/NDE actuelles utilisent la carte PR4 NILC Compton-y et son champ `FULL`. Le chemin de la carte doit être explicitement fourni ; le code ne choisit pas ni ne télécharge une carte par défaut.

1. Recherche de l'unique HDU HEALPix.
2. Lecture et validation de NSIDE, ORDERING et COORDSYS.
3. Prise en charge de RING/NESTED et des coordonnées galactiques ou équatoriales ; contrôle du nombre de pixels `12 × NSIDE²` et du stockage HEALPix implicite complet.
4. Lecture du champ `FULL` par défaut.
5. Conversion en NaN des valeurs non finies, du marqueur HEALPix UNSEEN et de BAD_DATA lorsqu'il est défini.
6. **Reprojection bilinéaire sur toute la grille radio native** avec conversion de coordonnées par la bibliothèque de reprojection.
7. Validité initiale : valeur finie ET footprint strictement positif.
8. Extraction de la même fenêtre centrale que la radio.
9. Même moyenne par blocs f × f.
10. Masque final `isfinite(planck_reduit)`.
11. Standardisation sur les statistiques du train ; pixels invalides remis à zéro.

La projection historique Planck commence donc sur toute la carte radio ; CHANDRA et LEGACY commencent directement sur la fenêtre radio utile. Le crop et la réduction finaux correspondent à la même grille.

La reprojection n'améliore pas la résolution instrumentale de Planck. Le code conserve les métadonnées de beam disponibles, mais ne déconvolue pas Planck et ne lisse pas automatiquement RAW/CHANDRA au beam Planck. Une image Planck 128 × 128 reste fortement corrélée spatialement à son échelle instrumentale.

Sources : [lecture et projection Planck](../src/clusterprep/planck.py#L43), [alignement et réduction](../src/clusterprep/planck.py#L145), [configuration Planck fold 0](../configs/de_nde_planck_5folds/fold_0.json).

## 5. Normalisation : formules et différences entre configurations

### 5.1 Percentiles et asinh pour radio, CHANDRA et LEGACY

Pour les pixels valides, on calcule p30 et p99, puis :

```text
u = clip((x − p30) / (p99 − p30 + 10⁻⁶), 0, 1)
x_normalisé = asinh(10 × u) / asinh(10)
```

Les calculs de quantiles utilisent les valeurs finies ; les calculs internes de la transformation utilisent des flottants de double précision, puis la sortie est en `float32`.

- Les valeurs ≤ p30 deviennent 0.
- Les valeurs ≥ p99 deviennent 1, à l'effet numérique du terme 10⁻⁶ près.
- L'asinh comprime la dynamique et donne davantage de poids relatif aux intensités intermédiaires.
- Les pixels invalides valent 0 en sortie.
- Une image sans pixel valide donne une image nulle.
- Une image constante donne zéro avec ses bornes calculées par image.

Cette transformation supprime une partie de l'information d'amplitude absolue, particulièrement lorsqu'elle est recalculée pour chaque image. Elle ne conserve pas le flux intégré. Le seuil inférieur est un percentile, pas un seuil de détection en nombre de sigma du bruit.

### 5.2 Portée « image » versus « train »

**Défaut logiciel :** p30/p99 sont recalculés séparément pour chaque image radio/X et chaque bande optique. Validation et test appliquent la même opération à leurs propres images ; aucune statistique populationnelle n'est ajustée sur ces splits.

**Si `scope="train"` :** les pixels valides des images d'entraînement, après alignement/réduction et avant augmentation/bruit, sont concaténés pour chaque modalité et chaque bande. Les percentiles sont calculés sur cet ensemble de pixels. Les mêmes bornes sont ensuite appliquées au train, à la validation et au test du fold. Ce n'est ni une moyenne des percentiles par amas, ni une pondération égale des amas : les images avec davantage de pixels couverts contribuent davantage.

Les statistiques sont recalculées à chaque fold et enregistrées avec les clés du train. Les caches physiques ne contiennent pas cette normalisation.

### 5.3 Standardisation Planck

```text
μ = moyenne de tous les pixels Planck valides du train
σ = écart-type populationnel de ces pixels
z = (y − μ) / σ
```

Si σ = 0, le dénominateur utilisé est 1. Les pixels invalides restent à zéro. Il n'y a par défaut ni clipping à [0,1], ni asinh sur Planck ; les valeurs négatives valides sont conservées après standardisation. Le masque distingue un pixel absent d'une valeur valide proche de la moyenne.

Si aucune valeur couverte n'existe dans le train pour une modalité à normaliser globalement, l'ajustement échoue. La présence d'une carte Planck ne transforme pas une erreur de lecture/projection en modalité silencieusement absente.

Attention aux noms : l'option Planck facultative `percentile_asinh` implémente une autre formule que `asinh_percentile` ; elle n'est pas le défaut décrit ici.

### 5.4 Ce que choisissent les configurations examinées

| Famille de configurations | Radio | CHANDRA | LEGACY | Planck |
|---|---|---|---|---|
| Radio seule, radio + CHANDRA | Asinh p30–p99 par image | Par image si sélectionné | — | — |
| Radio + LEGACY, radio + LEGACY + CHANDRA | Par image | Par image si sélectionné | Asinh p30–p99 sur le train, par bande | — |
| RAW + Planck, RAW + CHANDRA + Planck, suite DE/NDE actuelle | Asinh p30–p99 sur le train | Sur le train si sélectionné | — | Moyenne/écart-type du train |

Ajouter Planck **n'impose pas dans le code** une normalisation radio/X sur le train : ce sont les JSON de cette suite qui le demandent. De même, LEGACY a un défaut logiciel par image, mais les suites optiques examinées demandent explicitement une normalisation sur le train.

Sources : [normalisation élémentaire](../src/clusterprep/preprocessing.py#L83), [normalisation et ajustement des statistiques](../src/clusterprep/modalities.py#L40), [ajustement par fold](../src/clusterprep/training.py#L1293), [suite radio/optique](../configs/de_nde_radio_optical_5folds/fold_0.json), [suite RAW/T50kpcSUB + CHANDRA](../configs/de_nde_raw_t50kpcsub_chandra_5folds/fold_0.json), [suite optique/X](../configs/de_nde_radio_optical_xray_5folds/fold_0.json).

## 6. Les neuf parcours demandés

Dans les séquences ci-dessous, « géométrie radio » signifie lecture → fenêtre centrale → moyenne par blocs → WCS final. « Augmentation » signifie les opérations du train décrites en section 8 ; validation/test s'arrêtent au tenseur normalisé.

Les nombres de canaux correspondent au tenseur concaténé du dataset, avec masques activés, trois bandes LEGACY, et une seule source radio. Les modèles à plusieurs encodeurs peuvent consommer des branches séparées plutôt que ce tenseur complet.

### 6.1 RAW

1. Sélection du FITS RAW.
2. Géométrie radio RAW, sortie 128 × 128.
3. Normalisation asinh p30–p99 par image par défaut et dans les configurations radio seules examinées.
4. Pixels invalides à zéro.
5. Tenseur radio `float32`, puis augmentation du train.

**1 canal : RAW.** Aucun masque radio n'est ajouté au tenseur concaténé par le layout historique explicite.

### 6.2 RAW + CHANDRA

1. Géométrie radio RAW.
2. Lecture CHANDRA → reprojection sur la fenêtre RAW native → moyenne par blocs → masque X.
3. Normalisation de RAW et CHANDRA séparément, par image dans la suite RAW/T50kpcSUB + CHANDRA examinée.
4. Assemblage et augmentation synchronisée.

**3 canaux : RAW, CHANDRA, masque CHANDRA.** En mode multimodal optionnel, un CHANDRA absent produit deux canaux nuls ; en mode pair, l'amas sans fichier CHANDRA est exclu en amont ou refusé selon le point d'entrée.

### 6.3 RAW + CHANDRA + LEGACY

1. Géométrie radio RAW.
2. CHANDRA aligné/réduit sur RAW, avec son masque.
3. g/r/z alignés/réduits indépendamment sur RAW, avec trois masques.
4. Radio/X normalisés par image dans les suites examinées ; LEGACY normalisé avec les percentiles du train par bande dans les suites optiques.
5. Assemblage puis augmentation identique de toutes les modalités et masques.

**9 canaux : 1 radio + 1 X + 1 masque X + 3 optiques + 3 masques optiques.** L'ordre effectif dépend de la liste `modalities` : les JSON optique/X examinés utilisent radio, LEGACY, X.

### 6.4 RAW + LEGACY

1. Géométrie radio RAW.
2. Lecture g/r/z → reprojection sur RAW natif → réduction par blocs → trois masques.
3. Radio normalisé par image ; LEGACY normalisé par bande sur le train dans les suites optiques.
4. Assemblage puis augmentation synchronisée.

**7 canaux : RAW, g, r, z, masque g, masque r, masque z**, lorsque les modalités sont radio puis LEGACY.

### 6.5 T50kpcSUB

1. Sélection du FITS T50kpcSUB déjà existant.
2. Géométrie radio calculée à partir des dimensions et du WCS de T50kpcSUB.
3. Normalisation asinh p30–p99 par image, sauf configuration explicite contraire.
4. Pixels invalides à zéro, assemblage, augmentation du train.

**1 canal : T50kpcSUB**, exposé sous la clé interne `raw`. RAW n'est pas ajouté comme second canal. Aucun nouveau taper ou retrait de sources n'est exécuté ici.

### 6.6 T50kpcSUB + CHANDRA

1. Géométrie radio T50kpcSUB.
2. Reprojection CHANDRA sur la fenêtre T50kpcSUB native, puis mêmes blocs et masque X.
3. Normalisation indépendante par image dans la configuration T50kpcSUB + CHANDRA examinée.
4. Assemblage puis augmentation synchronisée.

**3 canaux : T50kpcSUB, CHANDRA, masque CHANDRA.** Le CHANDRA prétraité peut différer de celui du cas RAW + CHANDRA si les grilles radio diffèrent.

### 6.7 T50kpcSUB + CHANDRA + LEGACY

1. Géométrie radio T50kpcSUB.
2. CHANDRA aligné/réduit sur cette grille, avec masque X.
3. g/r/z alignés/réduits sur cette même grille, avec trois masques.
4. Par défaut logiciel : normalisation indépendante par image/bande pour toutes les intensités.
5. Si on reprend les paramètres des suites optiques examinées : LEGACY utilise les statistiques du train, radio/X restent par image.
6. Assemblage puis augmentation synchronisée.

**9 canaux.** Ce parcours est déduit de la composition des branches du code ; il ne faut pas lui attribuer automatiquement un preset d'expérience particulier. Il ne suppose pas que l'image CHANDRA/LEGACY prétraitée pour RAW est réutilisable telle quelle.

### 6.8 RAW + Planck

1. Géométrie radio RAW.
2. Lecture de la carte HEALPix Compton-y et nettoyage des pixels invalides.
3. Projection Planck sur toute la grille RAW native.
4. Crop identique à RAW, moyenne par les mêmes blocs, masque Planck.
5. Dans la suite Planck DE/NDE actuelle : percentiles radio ajustés sur le train et moyenne/écart-type Planck ajustés sur le train.
6. Application de ces statistiques à chaque split.
7. Assemblage puis augmentation synchronisée du train.

**3 canaux : RAW, Planck, masque Planck.** Avec les seuls défauts logiciels, la radio resterait normalisée par image ; Planck resterait standardisé sur le train.

### 6.9 RAW + Planck + CHANDRA

1. Géométrie radio RAW.
2. CHANDRA projeté sur la fenêtre RAW native, réduit et masqué.
3. Planck projeté sur la grille RAW native entière, puis crop, réduction et masque.
4. Dans la suite Planck DE/NDE actuelle : percentiles radio et X ajustés séparément sur le train ; moyenne/écart-type Planck ajustés sur le train.
5. Application aux splits et augmentation synchronisée du train.

**5 canaux : 1 radio + 1 X + son masque + 1 Planck + son masque.** Les configurations actuelles ordonnent les modalités radio, X, Planck. Le libellé « RAW + Planck + CHANDRA » ne change pas les opérations.

## 7. Modalités manquantes, couverture et ordre des canaux

Avec `mode="multimodal"` et `missing_modality_policy="optional"`, les amas sans CHANDRA ou LEGACY restent utilisables. Les images et masques absents sont nuls. Le dataset ne synthétise pas de signal de remplacement.

Un fichier présent mais sans couverture peut également donner un masque nul. Pour radio/X/LEGACY, l'indicateur de disponibilité est basé sur la présence du chemin, pas nécessairement sur un recouvrement non nul ; pour Planck, il est basé sur l'existence de pixels couverts.

Le mode `pair` ajoute des contraintes de présence. Sans Planck, le filtrage historique de ce mode exige CHANDRA, même avec certaines listes explicites de modalités. Il faut donc lire ensemble `mode`, `modalities` et `missing_modality_policy`. Avec Planck en mode pair, une projection sans aucun pixel valide déclenche une erreur.

La politique `required` filtre les modalités requises ; en mode multimodal, CHANDRA reste explicitement optionnel dans cette logique. Ce n'est pas un contrôle de seuil minimal de surface couverte.

L'ordre concaténé suit `modalities`. Pour chaque modalité auxiliaire, les plans d'intensité sont suivis des plans de masque. La radio n'ajoute pas de canal de masque dans ce layout, même si le dataset explicite expose une clé de masque radio séparée.

Sources : [sélection et layout](../src/clusterprep/modalities.py), [assemblage des échantillons](../src/clusterprep/dataset.py).

## 8. Augmentations réellement actives par défaut à l'entraînement

### 8.1 Rotations et retournements

Après normalisation :

- 12 rotations : 0°, 30°, 60°, …, 330°.
- Pour chaque angle : sans retournement ou avec retournement horizontal.
- Total : **24 variantes déterministes par amas** avec l'augmentation activée.
- Retournement horizontal d'abord, rotation ensuite.
- Interpolation **au plus proche voisin** pour ce chemin sans zoom.
- Taille conservée, sans expansion ; zones exposées remplies à zéro.
- Même transformation pour toutes les bandes et tous les masques.

Les 24 variantes sont représentées par les indices du dataset, et non par un angle arbitraire tiré à chaque chargement. La validation et le test utilisent l'image non augmentée. Il n'y a pas de nouvelle normalisation après rotation.

Le zoom est désactivé par défaut. L'ajout de bruit radio est également désactivé ; aucune estimation RMS destinée à cette augmentation n'est nécessaire dans le parcours par défaut. Si explicitement activé, le bruit est ajouté aux valeurs radio linéaires réduites avant normalisation et géométrie.

### 8.2 MixUp, après constitution des batches

Les réglages d'entraînement par défaut et les JSON Planck examinés activent aussi MixUp : probabilité de 0,5 par batch et paramètre α = 0,4.

```text
λ ~ Beta(0,4 ; 0,4)
entrée mélangée = λ × entrée + (1 − λ) × entrée permutée
```

La même permutation et le même λ sont appliqués aux tenseurs d'entrée du modèle, donc aussi aux masques lorsqu'ils font partie de ces entrées. Les masques peuvent alors devenir souples. Les cibles sont utilisées dans la perte pondérée correspondante.

MixUp est une augmentation d'entraînement après le preprocessing par image ; il n'est ni stocké dans les FITS, ni appliqué à la validation/test. Certains modèles avec scattering précalculé désactivent ou refusent MixUp ; ce n'est pas une étape inconditionnelle de tous les modèles.

Sources : [géométrie augmentée](../src/clusterprep/augmentation.py), [activation train seulement](../src/clusterprep/training.py#L1403), [MixUp](../src/clusterprep/training_core.py#L226).

## 9. Cache et ordre complet des opérations

Le cache disque conserve les tableaux **avant normalisation et augmentation**, avec valeurs physiques, NaN, WCS, masques et provenance. Les produits radio alternatifs disposent de chemins distincts. LEGACY possède un cache dépendant de la sélection des bandes et du HDU.

Planck utilise un cache de projection et un cache réduit ; leurs identités intègrent la source, le champ, la grille et les paramètres pertinents. Les données sont réutilisées uniquement si les contrôles de provenance et d'intégrité passent. Un cache invalide est reconstruit.

Ordre conceptuel :

```text
Inventaire / sélection du produit / éligibilité / splits par amas
→ lecture FITS
→ définition du crop et de la grille radio
→ alignement des modalités et réduction par blocs
→ masques de validité
→ cache physique déterministe
→ ajustement des statistiques sur le train, si scope=train
→ bruit radio optionnel, désactivé par défaut
→ normalisation / zéros hors couverture
→ tenseurs et canaux de masque
→ rotations/retournements du train
→ batch
→ MixUp du train si activé et compatible
→ modèle
```

Le calcul des statistiques train se fait sur les images propres, sans les variantes géométriques ni le bruit. Les transformations physiques déterministes peuvent être mises en cache pour plusieurs splits ; cela n'inclut pas d'apprentissage de statistiques sur la validation/test.

Sources : [cache physique](../src/clusterprep/processed.py), [cache Planck](../src/clusterprep/planck.py), [orchestration](../src/clusterprep/training.py).

## 10. Opérations absentes du chemin par défaut et limites d'interprétation

Le chemin historique n'ajoute pas de correction cosmologique, de mise à échelle commune en kpc/R500, d'homogénéisation des PSF RAW/T50kpcSUB, de conversion systématique Jy/beam → Jy/sr, de déconvolution, de segmentation de sources, ni de normalisation ImageNet. Il ne refait pas les traitements instrumentaux déjà incorporés aux FITS d'entrée.

Une branche `matched_resolution` existe, mais doit être explicitement sélectionnée. Elle constitue un autre protocole spatial et ne doit pas être confondue avec les neuf cas historiques décrits ici. LEGACY n'est pas accepté dans la combinaison de modalités du chemin matched d'entraînement inspecté. Voir [documentation du protocole matched](MATCHED_RESOLUTION.md).

Les différences scientifiquement importantes pour comparer les expériences sont donc :

1. Le produit radio choisi détermine la grille historique et peut changer le champ observé.
2. La réduction par blocs ne garantit pas une résolution instrumentale commune.
3. Les masques codent la validité finale, pas une fraction de couverture ou une exposition.
4. Les suites optiques et Planck remplacent certains défauts de normalisation ; cela fait partie du protocole expérimental.
5. Une normalisation par image ne préserve pas les amplitudes absolues entre amas ; une normalisation commune au train les conserve davantage, mais le clipping percentile continue de saturer les extrêmes.
6. Les réglages précis d'un run restent ceux de sa configuration sauvegardée : le nom de sa combinaison de modalités ne suffit pas à les déterminer.
