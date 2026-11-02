# RR et RH binaires : deux modèles, deux augmentations, cinq folds

Chaque fichier `fold_0.json` à `fold_4.json` regroupe désormais huit expériences :

| Bruit radio | Classifications | Modèles |
| --- | --- | --- |
| `gaussian_pixel` | `rr_binary`, `rh_binary` | `fusion`, `dual_encoder_cnn` |
| `beam_correlated` | `rr_binary`, `rh_binary` | `fusion`, `dual_encoder_cnn` |

Les quatre expériences originales sont conservées dans chaque variante, avec
les mêmes hyperparamètres. Le bruit radio est activé uniquement sur le train,
avec probabilité 0,5 et alpha uniforme dans [0 ; 0,5]. Les noms et sorties
permettent de distinguer les méthodes et les anciennes expériences sans bruit.

## Commande recommandée : gaussien puis corrélé

Depuis la racine du projet :

```sh
bash clusterprep/run_radio_noise_5folds.sh
```

Le [nouveau script](../../run_radio_noise_5folds.sh) utilise les
[configurations séparées par méthode](../radio_noise/) : d'abord les cinq folds
gaussiens, puis les cinq folds corrélés. Total : **40 entraînements**.
Pour afficher les commandes seulement, ajouter `--dry-run`. Pour sélectionner
les folds, ajouter leurs indices, par exemple `2 3 4`.
Les sorties sont sous `results/radio_noise/<méthode>/fold_N/`.

## Variante : lancer les JSON regroupés par fold

```sh
bash clusterprep/run_binary_subtypes_5folds.sh
```

Ce [script](../../run_binary_subtypes_5folds.sh) lance les JSON de ce dossier :
huit expériences sur le fold 0, puis huit sur le fold 1, etc. Il entraîne les
**mêmes 40 expériences**, dans un autre ordre, et écrit sous
`results/binary_subtypes_multimodal_5folds/radio_noise/fold_N/`.
Il n'est pas nécessaire de lancer les deux scripts.

Les deux scripts se placent à la racine du projet, utilisent `.venv/bin/python`
(surcharge par `PYTHON_BIN`), acceptent `--dry-run` et des indices de folds,
et s'arrêtent à la première erreur. Relancer crée de nouveaux runs, sans reprise
automatique des checkpoints. La préparation ne lance aucun entraînement.

## Protocole conservé

- Mode `multimodal`, avec ou sans CHANDRA et masque de modalité.
- Sorties binaires `[non_RR, RR]` et `[non_RH, RH]`, cible scalaire.
- `candidate_policy: positive` : cRR/cRH assimilés à RR/RH.
- Cinq folds de test, seed des splits 42 ; validation 20 %, train environ 60 %.
- `cross_validation: null` : pas de CV interne supplémentaire.
- 200 epochs maximum, batch 16, learning rate `5e-5`, weight decay `0.1`.
- Early stopping sur `val_loss`, patience 50, OneCycle, MixUp, pondération des
  classes et augmentations géométriques conservés.
- Images 128×128, `width: 16`, périphérique `mps`, cache `processed/` partagé.
- Dossiers d'annotations découverts automatiquement ; filtrage binaire existant.
- Comparaison des architectures séparée par tâche, `comparison_cohort: all`.

Une [baseline sans bruit supplémentaire](../radio_noise/baseline/) est disponible
avec les mêmes modèles, tâches et paramètres ; elle n'est pas lancée par les
scripts ci-dessus.
