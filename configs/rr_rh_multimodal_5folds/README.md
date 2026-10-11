# RR/RH multimodal — cinq folds

Chaque `fold_0.json` à `fold_4.json` entraîne les six modèles : cnn, image_cnn,
fusion, dual_encoder_cnn, dual_ssn et dual_encoder_ssn. Les six architectures
partagent les mêmes splits dans un fold. L'ensemble représente 30 entraînements.

Depuis la racine `PH_Project_I`, lancer les cinq folds séquentiellement :

```sh
for fold in 0 1 2 3 4; do
  PYTHONPATH=clusterprep/src .venv/bin/python -m clusterprep \
    "clusterprep/configs/rr_rh_multimodal_5folds/fold_${fold}.json" || break
done
```

Pour lancer seulement le fold 0 :

```sh
PYTHONPATH=clusterprep/src .venv/bin/python -m clusterprep \
  clusterprep/configs/rr_rh_multimodal_5folds/fold_0.json
```

`n_folds: 5` définit le découpage ; chaque invocation entraîne seulement le
`fold_index` du fichier. Les chemins sont relatifs au dossier de lancement,
et non au dossier contenant le JSON.

Paramètres repris de `configs/training_config.json` : 200 epochs maximum,
batch 16, learning rate 5e-5, patience 10, augmentations et mixup activés,
device MPS. Les seuils RR/RH sont fixés à 0.5. Pour une machine CUDA ou CPU,
modifier `device` dans les expériences. Les modèles SSN nécessitent Kymatio.

L'inventaire inclut DE, NDE, RR, RH, cRR, cRH. Les DE seuls sont exclus de
la tâche ; les NDE donnent [0,0] et les candidats sont considérés positifs.
Le mode multimodal conserve les amas sans CHANDRA, avec le masque de modalité.
La comparaison porte sur tous les amas de test (`comparison_cohort: all`).

La seed des splits reste 42 pour les cinq fichiers, afin que les folds de
test soient disjoints. Avec `val_size: 0.2`, les parts visées sont environ
60 % train, 20 % validation et 20 % test, avec les arrondis de stratification.

Sorties : `results/rr_rh_multimodal_5folds/fold_N/`. Chaque entraînement crée
son propre sous-dossier, contenant notamment les prédictions, métriques et
`split_distribution.json`. Les caches sont partagés sous `processed/`.
Relancer la commande crée de nouveaux runs ; ce n'est pas une reprise automatique.

Validation préalable des configurations et des splits du catalogue local :
`results/rr_rh_multimodal_5folds_validation.json`.
