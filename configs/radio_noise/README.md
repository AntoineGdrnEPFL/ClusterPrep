# RR et RH binaires — comparaison des augmentations radio

Chaque JSON reprend les quatre expériences de la suite binaire multimodale :

| Classification | Modèles |
| --- | --- |
| `rr_binary` : non_RR / RR | `fusion`, `dual_encoder_cnn` |
| `rh_binary` : non_RH / RH | `fusion`, `dual_encoder_cnn` |

Les trois dossiers [baseline](baseline/), [gaussian_pixel](gaussian_pixel/) et
[beam_correlated](beam_correlated/) contiennent chacun les cinq folds, quatre
expériences par fichier. Les modèles, tâches, mode multimodal, splits, seed 42 et
hyperparamètres sont identiques, notamment patience 50, 200 epochs maximum,
`candidate_policy: positive`, MixUp et les augmentations géométriques existantes.
Seuls le bruit radio, les noms de runs et les répertoires de sortie diffèrent.
Les anciens exemples multilabel `rr_rh` à une seule architecture sont remplacés.

Le bruit est désactivé pour baseline. Les deux autres méthodes sont activées sur
le train seulement, avec probabilité 0,5 et alpha uniforme dans [0 ; 0,5].
Validation/test, CHANDRA et masques ne reçoivent pas ce bruit.

## Lancement séquentiel

Depuis la racine du projet :

```sh
bash clusterprep/run_radio_noise_5folds.sh
```

Le [script](../../run_radio_noise_5folds.sh) lance **les cinq folds gaussian_pixel,
puis les cinq folds beam_correlated** : 10 configurations, **40 entraînements**,
sans parallélisme. Il s'arrête à la première erreur. La baseline n'est pas lancée.
Le script retrouve la racine du projet, définit PYTHONPATH et utilise `.venv/bin/python`
par défaut ; `PYTHON_BIN` permet de sélectionner un autre interpréteur.

```sh
# Afficher les commandes sans entraîner
bash clusterprep/run_radio_noise_5folds.sh --dry-run
# Limiter aux folds 2, 3 et 4, pour les deux méthodes
bash clusterprep/run_radio_noise_5folds.sh 2 3 4
```

Les sorties sont séparées sous `results/radio_noise/<méthode>/fold_N/`.
Une relance crée de nouveaux runs et ne reprend pas automatiquement un checkpoint.
Pour la baseline facultative (20 entraînements supplémentaires) :

```sh
for fold in 0 1 2 3 4; do
  PYTHONPATH=clusterprep/src .venv/bin/python -m clusterprep \
    "clusterprep/configs/radio_noise/baseline/fold_${fold}.json" || break
done
```

## Configurations binaires regroupées

Les [JSON binaires](../binary_subtypes_multimodal_5folds/) contiennent aussi les
mêmes variantes, regroupées en huit expériences par fold (quatre gaussiennes,
puis quatre corrélées). Leur ancien script reste utilisable, mais ordonne les
runs par fold et écrit dans un dossier distinct. Ce sont **deux façons de lancer
les mêmes expériences** : inutile de lancer les deux scripts pour une comparaison.
