# RAW et T50kpcSUB avec Chandra — DualEncoderCNN DE/NDE

Les cinq JSON [fold_0](fold_0.json), [fold_1](fold_1.json),
[fold_2](fold_2.json), [fold_3](fold_3.json) et [fold_4](fold_4.json)
contiennent chacun deux expériences, dans cet ordre :

1. `radio_source: RAW` + Chandra.
2. `radio_source: T50kpcSUB` + Chandra.

Architecture `dual_encoder_cnn`, tâche `de_nde`, mode `multimodal`, modalités
`["radio", "xray"]`. Les expériences ne diffèrent que par leur nom et le produit
radio. Les hyperparamètres reprennent la suite radio/optique existante :
200 époques maximum, patience 50, batch 16, images 128², width 16, fusion 64,
learning rate 5e-5, weight decay 0.1, OneCycleLR, MixUp, pondération des classes,
rotations et retournements. Aucun zoom ni bruit radio ajouté. Normalisation
historique par image ; aucun transfert de poids. Device MPS ; compilation `auto`
(active sur MPS/CUDA, désactivée sur CPU).

## Population et folds

Le produit T50kpcSUB manque pour certains amas. Le lanceur sélectionne donc
**l'intersection RAW/T50kpcSUB avant les splits**, identique pour les deux modèles.
Il enregistre les IDs inclus, les IDs exclus avec leur motif, la couverture et
les splits de chaque fold dans `results/de_nde_raw_t50kpcsub_chandra_5folds/population_audits/`.
Le catalogue original et les FITS ne sont pas modifiés.

Chandra reste **optionnel**, conformément au mode multimodal : lorsqu'il manque,
son canal et son masque sont nuls. Il n'y a pas de sélection limitée aux seules
paires. Les modèles utilisent les mêmes amas, les mêmes labels et les mêmes splits.
La radio reste strictement requise (`missing_radio_policy: error`) après la
sélection commune ; aucun remplacement d'un tapered absent par RAW.

Cinq folds externes, graine de split et modèle 42, validation `val_size: 0.2`
selon le splitter existant. Chaque amas commun apparaît dans le test d'un seul
fold. Pas de CV interne ni réentraînement additionnel : **10 entraînements**.
Le champ angulaire et le faisceau peuvent différer entre RAW et T50kpcSUB à
résolution de sortie identique ; cette comparaison ne les homogénéise pas.

## Lancement

Depuis la racine PH_Project_I :

```sh
# Vérifier les configurations, sans lire les FITS ni entraîner
bash clusterprep/run_de_nde_raw_t50kpcsub_chandra_5folds.sh --dry-run

# Auditer la population réelle et les splits, sans entraîner
bash clusterprep/run_de_nde_raw_t50kpcsub_chandra_5folds.sh --check-data

# Lancer les dix entraînements
bash clusterprep/run_de_nde_raw_t50kpcsub_chandra_5folds.sh

# Lancer seulement les folds 3 et 4 (quatre entraînements)
bash clusterprep/run_de_nde_raw_t50kpcsub_chandra_5folds.sh 3 4
```

Utiliser ce [lanceur](../../run_de_nde_raw_t50kpcsub_chandra_5folds.sh) pour la
comparaison : il applique la population commune avant d'appeler l'API
`run_experiments`. Un lancement direct des JSON par `python -m clusterprep`
ne réalise pas cette sélection et échouerait sur les produits radio absents.
L'inventaire vérifie les contenus FITS et peut prendre un moment ; il n'est lu
qu'une fois par invocation pour tous les folds.

Le script retrouve la racine du projet et utilise `.venv/bin/python` si disponible ;
`PYTHON_BIN` permet de changer d'interpréteur. Les entraînements sont séquentiels
et s'arrêtent à la première erreur. Une relance crée de nouveaux runs, sans reprise
automatique de checkpoint. Les JSON permettent de changer le device ; conserver
les mêmes paramètres pour les deux variantes.

Résultats séparés par fold sous `results/de_nde_raw_t50kpcsub_chandra_5folds/fold_N/`,
avec un CSV de comparaison sur toute la cohorte test commune après chaque paire.
Chaque run reçoit aussi le rapport de population commune et l'audit radio.
