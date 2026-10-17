# Transfert modulaire, indépendant de la tâche

Le [nouvel exemple JSON](examples/transfer_modular.json) initialise les deux
encodeurs d'un `dual_encoder_cnn` depuis deux expériences différentes. Adapter
les checkpoints et le dossier de données avant de lancer, depuis `PH_Project_I` :

```sh
PYTHONPATH=clusterprep/src .venv/bin/python -m clusterprep clusterprep/examples/transfer_modular.json
```

Le même schéma fonctionne pour `de_nde` et `rr_rh`, dans les deux sens. La tâche,
la loss, le nombre de sorties et les métriques restent ceux de la cible.
Les options usuelles (`model`, `model_params`, `learning_rate`, etc.) gardent leur
format existant. Le moteur est dans [modular_transfer.py](src/clusterprep/modular_transfer.py).

## Configuration

`transfer` accepte `enabled`, `sources`, `mappings` et `defaults`.
Sans `enabled: true`, aucun checkpoint n'est consulté. Le transfert absent ou
désactivé conserve l'entraînement habituel.

- `sources` associe un nom à `{"checkpoint": "…/best.pt"}` et, si nécessaire,
  `"config_path": "…/source_config.json"`. Les chemins relatifs sont résolus
  depuis le répertoire de travail, comme les chemins des suites historiques.
- `mappings` contient des objets `{"source": "nom", "from": "module_source",
  "to": "module_cible"}` dans l'ordre souhaité. Aucun module n'est implicite.
  Même compatible, le classifieur n'est jamais copié sans mapping explicite.
- Les règles locales remplacent les règles de `defaults`. Les valeurs par défaut
  sont `compatibility: strict`, `on_missing: warn`, `on_incompatible: skip`,
  `on_conflict: error`.
- `on_missing` et `on_incompatible` acceptent `error`, `warn` (ignorer et avertir)
  et `skip` (ignorer et consigner le motif). Un checkpoint invalide constitue une
  erreur fatale. Si aucun mapping n'est applicable, le run échoue.
- Les collisions sont toujours fatales, y compris entre alias, modules imbriqués
  ou paramètres/buffers partageant du stockage. `overwrite` n'est pas proposé.

Les erreurs de configuration sont détectées sans charger les checkpoints.
Les erreurs de reconstruction ou de compatibilité sont détectées à la préparation
et conservées dans un rapport `TransferError.report`.

## Modules disponibles

Les noms logiques existants sont `radio`, `xray` et `head`, selon le modèle.
Les alias `radio_encoder`, `xray_encoder` et `classifier` sont aussi acceptés.
Les vrais chemins de sous-modules PyTorch sont utilisables (`encoder`,
`chandra_encoder`, `encoder.to_latent`, `classifier.0`, etc.). La résolution
privilégie un nom logique exact, puis un chemin réel, puis un alias.
`DataParallel` et `DistributedDataParallel` sont déballés pour la résolution.

La fusion fonctionnelle dans `forward` n'est pas un module transférable.
Un modèle externe qui possède un vrai module `fusion` peut le sélectionner.
Le registre de groupes pour l'entraînement doit rester une partition exhaustive
sans recouvrement ; les mappings d'initialisation peuvent viser des sous-modules.

La vérification stricte compare récursivement types et ordre des sous-modules,
options du forward (stride, padding, activation, etc.), noms/formes/dtypes des
paramètres et buffers, persistance des buffers et partage des tenseurs. Elle
ne se limite pas aux shapes. Les hooks et les `forward` remplacés sur une instance
sont refusés. Les modules PyTorch aux options non vérifiables sont refusés aussi.

Un module personnalisé doit définir directement une méthode
`transfer_signature()` retournant une valeur JSON avec une version d'architecture
et les options influençant son forward. Modifier sa version dès que l'ordre ou
le calcul change. Les sous-modules sont contrôlés récursivement, même avec cette
signature ; un sous-module personnalisé doit donc lui aussi déclarer la sienne.
Les encodeurs ImageCNN et DualSSN et leur attention/scattering sont pris en charge.
Une signature ne doit pas dépendre des poids entraînés ni du device.

## Checkpoints et reproductibilité

Chaque source est reconstruite via le registre/factory existant avec **sa propre**
configuration et son nombre de classes. Les sources ne déclenchent jamais leur
ancien plan de transfert. Leurs états d'optimizer/scheduler ne sont pas importés.
Le RNG CPU PyTorch, Python et NumPy est préservé pendant leur construction.
Un checkpoint est désérialisé une seule fois, même si plusieurs mappings ou noms
de source l'utilisent. La reconstruction est réutilisée par couple checkpoint/config.

La configuration vient, dans l'ordre :

1. du `config_path` explicite, s'il est fourni ;
2. de `config` dans le checkpoint ;
3. du fichier `config.json` adjacent, convention des anciens runs.

Le JSON source est une configuration d'expérience ou l'objet de métadonnées
`{"config": {...}, "class_names": [...]}`. Il doit contenir au minimum `model`,
`task`, `mode`, `size` et les paramètres nécessaires à son architecture réelle.
Pour les anciens runs sans `task`, les `class_names` enregistrées permettent
de restaurer cette information si elles correspondent sans ambiguïté à une tâche
connue (`DE`/`NDE` ou `RR`/`RH`). Cette migration apparaît dans
`metadata_migrations` du rapport ; des classes absentes ou inconnues nécessitent
toujours un `config_path` avec une tâche explicite.
Une suite contenant `experiments` doit être extraite en une configuration source
unique. Les poids sans ces métadonnées sont refusés ; l'architecture n'est pas
inférée à partir des shapes. Les anciens noms d'ImageCNN suivent uniquement la
migration explicite déjà supportée ; un checkpoint incomplet reste refusé.
Les métadonnées anciennes décrivent l'architecture selon la version installée
du constructeur : conserver le code correspondant au run source.

Les nouveaux checkpoints embarquent aussi les métadonnées de preprocessing et le
rapport de transfert. Un ancien preprocessing absent est indiqué par `null` dans
la provenance ; aucune transformation des poids n'est devinée. Les paramètres de
preprocessing source sont consignés, sans remplacer ceux de la cible.

Le run enregistre sa configuration effective et `transfer.json`, avec les chemins
absolus, SHA-256, configurations sources, résultats et motifs par mapping, ainsi
que le nombre de paramètres copiés. Le même rapport figure dans les métriques et
les checkpoints. Les erreurs de transfert produisent également `transfer.json`
et un `status.json` en échec, avant tout entraînement. Les modules ignorés gardent
leur initialisation ; toute erreur fatale de prévalidation laisse la cible intacte.

## Entraînement et reprise

L'ordre est : construction de la cible → transfert → `Trainer` → phases et
optimizers → entraînement/évaluation. Sans phases, l'optimizer habituel est créé.

Les phases se configurent dans `training_phases`, **à côté** de `transfer`, selon
la convention de champs d'entraînement à plat du projet. Elles réutilisent le
moteur existant : les groupes absents de `learning_rates` sont gelés, leurs
BatchNorm restent en mode évaluation, puis un nouvel optimizer est créé pour
chaque phase. Les noms de groupes sont `radio`, `xray`, `head`. Les phases peuvent
également être utilisées sans aucun transfert. Le rapport des phases est sauvegardé
dans `training_phases.json` ; elles ne modifient pas le rapport d'initialisation.

Il n'existe pas de resume complet dans le runner actuel. `best.pt` est un checkpoint
de sélection sur validation, pas une promesse de reprise exacte du dernier batch.
L'API `initialize_transfer(..., resume=True)` refuse explicitement un transfert
activé. Une future reprise devra restaurer ses propres poids et états sans appeler
l'initialisation. Le transfert n'introduit aucune règle implicite de reprise.

## Compatibilité et migration

Les JSON historiques avec `source_experiment`, `module_prefixes` et `phases`
restent sur le [parcours historique](TRANSFER_LEARNING.md), y compris ses règles
DE/NDE → RR/RH et ses splits. Ne pas mélanger les deux schémas.

Pour convertir volontairement une expérience historique :

1. Déclarer `source_experiment/best.pt` dans `sources` et `enabled: true`.
2. Convertir chaque `module_prefixes[cible] = prefixe` en mapping explicite.
3. Déplacer `transfer.phases` vers `training_phases`.
4. Choisir les splits de la cible avec les options ordinaires du runner ou fournir
   explicitement le résultat de `resolve_transfer_splits` lorsque souhaité. Le
   nouveau moteur ne réutilise ni n'audite automatiquement les splits sources.

Le wizard `createconfig` propose le choix **3. Transfert modulaire** pour les deux
tâches. Le choix historique reste disponible pour RR/RH. Les phases indépendantes
sont saisies dans le champ JSON `training_phases`.

L'[audit préalable](MODULAR_TRANSFER_AUDIT.md) décrit les composants réutilisés.
Les [tests du moteur](tests/test_modular_transfer.py) complètent les tests
historiques de transfert, d'entraînement et de configuration.
