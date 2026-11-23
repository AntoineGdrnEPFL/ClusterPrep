# Planck PR4 — bilan de validation du 16 novembre 2026

Implémentation et audit : [PLANCK_PR4.md](PLANCK_PR4.md).

## Résultats

- Tests Planck spécifiques : 15 réussites dans la version finale. Reprojection réelle d’un FITS HEALPix synthétique, conversion galactique→équatoriale comparée à un champ analytique, cache réutilisé/reconstruit/forcé, changement de source/WCS/forme, cache vide, métadonnées invalides, NaN/UNSEEN, WCS céleste extraite d’un FITS 4D, crops, rotations/flips/zoom et masques, normalisations, absence de CHANDRA, refus des paires sans couverture Planck, RAW/T50kpcSUB matched, checkpoints et transfert compatible.
- Lot ciblé de non-régression : **199 passed, 28 subtests passed** (avant les quatre derniers tests de robustesse Planck, qui passent séparément).
- Suite complète après corrections : **665 passed, 17 failed, 1 skipped, 33 subtests passed**. Les 17 échecs sont préexistants, détaillés ci-dessous. Quatre tests de robustesse supplémentaires ont été ajoutés et validés ensuite dans le lot Planck final.
- Les **40 configurations** ont passé `run_config_suite(..., dry_run=True)` : mêmes membres et effectifs que les cinq partitions existantes, sans lecture du test externe.
- **8 smoke tests réels terminés**, une époque chacun : A/B/C/D × historique/matched_resolution. Quatre objets train, deux validation, images 32 × 32, CPU, deux batches d’entraînement. Les checkpoints `best.pt` sont présents et tous les statuts sont `completed`.
- Objets d’entraînement : PSZ2G071.21+28.86 (RR+RH, sans CHANDRA), PSZ2G023.17+86.71 (RH, avec CHANDRA), PSZ2G045.13+67.78 (NDE, sans CHANDRA), PSZ2G045.87+57.70 (RH, sans CHANDRA). Validation : PSZ2G053.53+59.52 et PSZ2G056.77+36.32, provenant uniquement du split validation.
- Les 13 objets du test externe sont réservés et ne sont pas chargés dans les smoke tests. Les statistiques sont ajustées sur les quatre objets train seulement. Les avertissements d’AUC indéfinie sur la minuscule validation sont attendus ; ces scores ne mesurent pas l’apport scientifique de Planck.
- Cache PR4 vérifié deux fois avec le résultat `reused` sur le FITS radio réel **2304 × 2304** de PSZ2G023.17+86.71.
- Figure de contrôle générée et inspectée visuellement ; version finale avec crop 128 × 128. Les neuf panneaux montrent l’alignement WCS, la couverture X-ray partielle, les contours et le champ Planck large.
- `git diff --check` sans erreur.

Résultats réels : [résumé JSON](../../outputs/planck_smoke/summary.json) et [figure d’alignement](../../outputs/planck_smoke/alignment.png). Ces fichiers restent dans les sorties du workspace, hors dépôt de code.

## Commandes reproductibles

Depuis la racine du workspace :

```sh
.venv/bin/python -m pytest clusterprep/tests/test_planck.py -q

.venv/bin/python -m pytest \
  clusterprep/tests/test_planck.py \
  clusterprep/tests/test_models.py \
  clusterprep/tests/test_dual_encoder_modalities.py \
  clusterprep/tests/test_mode_semantics.py \
  clusterprep/tests/test_matched_dataset.py \
  clusterprep/tests/test_matched_evaluation.py \
  clusterprep/tests/test_legacy.py \
  clusterprep/tests/test_training.py \
  clusterprep/tests/test_cross_validation.py \
  clusterprep/tests/test_modular_transfer.py -q

.venv/bin/python -m pytest clusterprep/tests -q --tb=short

.venv/bin/python -m clusterprep \
  clusterprep/configs/planck/matched_resolution/C/fold_0.json --dry-run

# Huit entraînements très courts, pas les entraînements complets :
.venv/bin/python clusterprep/examples/planck_smoke.py
```

La commande complète conserve les 17 échecs préexistants décrits ci-dessous ; elle n’est pas déclarée verte.

## Échecs préexistants isolés

| Nombre | Origine |
|---|---|
| 11 | Les tests de `matched_runner` et `matched_config_cli` attendent `patience=10`, alors que les JSON locaux préexistants spécifient 50. Le runner original extrait de HEAD renvoie également 50. Les hyperparamètres n’ont pas été changés pour satisfaire ces assertions. |
| 1 | Ancienne assertion de `test_audit` attendant une exception pour CHANDRA absent en `pair` ; le code antérieur filtre déjà cette entrée. |
| 2 | Configurations d’exemple `rr_binary.json` et `rh_binary.json` absentes du workspace. |
| 1 | Smoke test notebook cherchant un FITS synthétique prétraité absent (`synthetic-36`). |
| 2 | Assertions de visualisation obsolètes : nombre d’axes incluant les colorbars, et titres anglais/français. |

Les six échecs hors patience ont été reproduits en exécutant les tests contre les sources originales extraites de HEAD dans un répertoire temporaire séparé (`PYTHONPATH`), sans modifier le checkout. Pour les onze échecs de patience, le runner original a été évalué directement sur les configurations préexistantes et confirme la valeur 50.

## Portée restante

Aucun entraînement complet ni comparaison scientifique des performances n’a été lancé. La mémoire/rapidité d’une campagne de 40 entraînements, l’usage multi-worker et l’effet du masque scientifique externe Planck restent à évaluer. Les smoke tests réels couvrent six objets et les tests synthétiques couvrent les cas limites ; ils ne constituent pas un audit de tous les FITS du catalogue.
