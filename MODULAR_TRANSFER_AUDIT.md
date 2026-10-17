# Audit préalable — transfert modulaire

Audit effectué avant les modifications du code.

- `training.py` : `ExperimentConfig` valide les dictionnaires JSON ; `_run`
  construit le modèle puis initialise les poids avant de créer `Trainer`.
  `Trainer._checkpoint` sauvegarde poids, config, classes et états d'entraînement.
- `models.py` : `build_model` et le registre reconstruisent les architectures.
  Tous les modèles exposent `transfer_modules()` (`radio`, `xray`, `head`).
  La fusion est souvent une opération fonctionnelle dans `forward`, sans module
  autonome ; elle ne peut donc pas être copiée comme un module.
- `transfer.py` : transfert historique mono-source DE/NDE → RR/RH,
  validation des groupes, phases et optimizers discriminatifs déjà disponibles.
  Les contrôles de poids seuls ne prouvent pas la compatibilité du forward.
- `experiment_loader.py` : reprise des splits et audit de fuite historiques,
  à préserver pour les configurations historiques.
- `training_core.py`, `metrics.py` : la tâche cible sélectionne loss et métriques.
- Aucun mécanisme de resume complet n'est implémenté. Charger des poids ne
  constitue pas une reprise des états d'entraînement.

## Compatibilités et décisions

- Conserver les configurations JSON historiques et leur parcours, ainsi que
  les expériences sans transfert. Aucun changement global de pipeline.
- Ajouter un schéma `enabled/defaults/sources/mappings`, indépendant des tâches,
  des splits et des phases. Garder les champs d'entraînement à plat, convention
  existante ; ajouter `training_phases` pour réutiliser le moteur de phases.
- Réutiliser les noms logiques existants et les chemins PyTorch ; fournir
  `radio_encoder`, `xray_encoder`, `classifier` comme alias si non ambigus.
- Reconstruire la source avec sa propre configuration via `build_model`.
  Utiliser les métadonnées du checkpoint, le `config.json` historique adjacent,
  ou un `config_path` explicite. Refuser les poids sans métadonnées suffisantes.
- Signatures explicites/versionnées pour les modules personnalisés ; contrôle
  récursif des modules standards, paramètres, buffers et attributs du forward.
- Prévalider les collisions et incompatibilités fatales avant toute copie.
  Seule la politique de conflit `error` est prise en charge initialement.
- Sauvegarder les rapports, y compris lors des échecs. Ne pas importer les
  états optimizer/scheduler ; ne pas introduire de fonctionnalité de resume.
