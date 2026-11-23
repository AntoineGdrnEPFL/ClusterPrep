#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
  echo "Usage : $0 [--dry-run | --check-data] [0 1 2 3 4]"
  echo "DualEncoderCNN DE/NDE multimodal : RAW + Chandra puis T50kpcSUB + Chandra."
  echo "Sans indice : cinq folds, dix entraînements séquentiels, sur la population radio commune."
  echo "--dry-run : valider les JSON sans lire les FITS ni entraîner."
  echo "--check-data : auditer la population et vérifier les splits sans entraîner."
  echo "PYTHON_BIN permet de choisir Python ; device est défini dans les JSON (MPS par défaut)."
}

ACTION=run
folds=()
for arg in "$@"; do
  case "$arg" in
    --dry-run|--check-data)
      if [[ "$ACTION" != run ]]; then
        echo "Choisir une seule option --dry-run ou --check-data." >&2; exit 2
      fi
      ACTION="$arg" ;;
    -h|--help) usage; exit 0 ;;
    0|1|2|3|4)
      for selected in "${folds[@]:-}"; do
        if [[ "$selected" == "$arg" ]]; then
          echo "Fold indiqué plusieurs fois : $arg" >&2; exit 2
        fi
      done
      folds+=("$arg") ;;
    *) echo "Argument invalide : $arg" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ ${#folds[@]} -eq 0 ]]; then folds=(0 1 2 3 4); fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Interpréteur Python introuvable : $PYTHON_BIN" >&2; exit 1
fi
export PYTHONPATH="$PROJECT_ROOT/clusterprep/src${PYTHONPATH:+:$PYTHONPATH}"

# Un seul inventaire FITS pour toute la suite. La sélection commune est explicite,
# auditée et appliquée AVANT tout split, sans fallback radio ni modification du catalogue.
"$PYTHON_BIN" - "$ACTION" "${folds[@]}" <<'PY'
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from clusterprep.io import get_data_info_dict, select_radio_source
from clusterprep.splits import split_data_keys
from clusterprep.training import ExperimentConfig, run_experiments, compare_experiments

action, *folds = sys.argv[1:]
config_root = Path('clusterprep/configs/de_nde_raw_t50kpcsub_chandra_5folds')
suites = []
for fold in folds:
    path = config_root / f'fold_{fold}.json'
    spec = json.loads(path.read_text())
    configs = [ExperimentConfig(**item) for item in spec['experiments']]
    if [c.radio_source for c in configs] != ['RAW', 'T50kpcSUB']:
        raise ValueError(f'{path}: deux expériences RAW puis T50kpcSUB requises')
    if any(c.model != 'dual_encoder_cnn' or c.mode != 'multimodal' or c.task != 'de_nde'
           or c.modalities != ['radio', 'xray'] or c.missing_modality_policy != 'optional'
           or c.missing_radio_policy != 'error'
           or (c.cross_validation is not None and c.cross_validation.enabled) for c in configs):
        raise ValueError(f'{path}: protocole DualEncoderCNN multimodal DE/NDE incohérent')
    comparable = [{k: v for k, v in item.items() if k not in {'name', 'radio_source'}}
                  for item in spec['experiments']]
    if comparable[0] != comparable[1]:
        raise ValueError(f'{path}: hyperparamètres différents entre les deux produits')
    if spec['split']['fold_index'] != int(fold) or spec['split']['n_folds'] != 5:
        raise ValueError(f'{path}: indice de fold incohérent')
    if suites and (spec['data_dir'], spec['class_names']) != (suites[0][1]['data_dir'], suites[0][1]['class_names']):
        raise ValueError('Toutes les configurations doivent utiliser le même catalogue et les mêmes classes')
    suites.append((path, spec, configs))
    print(f"Fold {fold}: {path} — RAW + Chandra puis T50kpcSUB + Chandra, device={configs[0].device}", flush=True)
if action == '--dry-run':
    print(f'{2 * len(suites)} entraînements prévus ; aucun FITS lu, aucun entraînement lancé.')
    sys.exit(0)

info, inventory = get_data_info_dict(suites[0][1]['data_dir'], suites[0][1]['class_names'], return_audit=True)
sources = ['RAW', 'T50kpcSUB']
common, excluded = {}, []
for key, item in info.items():
    missing = [source for source in sources
               if not item.get('radio_paths', {}).get(source)
               or not Path(item['radio_paths'][source]).is_file()]
    if missing:
        excluded.append({'key': key, 'cluster_id': item['cluster_id'], 'missing_radio_sources': missing})
    else:
        common[key] = item
if not common:
    raise ValueError('Population commune RAW/T50kpcSUB vide')
for source in sources:
    select_radio_source(common, source)  # Disponibilité stricte, aucune substitution.

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
audit_dir = Path(suites[0][1]['output_dir']).parent / 'population_audits' / stamp
audit_dir.mkdir(parents=True, exist_ok=False)
report = {
    'sources': sources, 'selection': 'intersection des produits radio avant les splits',
    'chandra_policy': 'optional: masque de présence/couverture, aucune exclusion',
    'catalog_count': len(info), 'included_count': len(common),
    'included': sorted(common), 'included_cluster_ids': sorted(v['cluster_id'] for v in common.values()),
    'with_chandra': sum(bool(v.get('chandra_path')) for v in common.values()),
    'excluded': excluded, 'configurations': [str(p) for p, _, _ in suites],
}
(audit_dir / 'common_population.json').write_text(json.dumps(report, indent=2))
(audit_dir / 'radio_inventory.json').write_text(json.dumps(inventory, indent=2))
print(f"Population commune : {len(common)}/{len(info)} amas, {report['with_chandra']} avec Chandra ; {len(excluded)} exclus. Audit : {audit_dir}", flush=True)

# Vérifier tous les folds demandés avant le premier entraînement.
for path, spec, configs in suites:
    options = dict(spec['split'])
    options['seed'] = options.pop('split_seed')
    splits = split_data_keys(common, mode='raw', task='de_nde', **options)
    manifest = dict(zip(('train', 'validation', 'test'), splits))
    (audit_dir / f"fold_{spec['split']['fold_index']}_splits.json").write_text(json.dumps(manifest, indent=2))
if action == '--check-data':
    print('Inventaire et splits vérifiés ; aucun entraînement lancé.')
    sys.exit(0)

for path, spec, configs in suites:
    print(f"Lancement {path}", flush=True)
    runs = run_experiments(common, configs, spec['output_dir'],
                           class_names=spec['class_names'], **spec['split'])
    for run in runs:
        (run / 'common_population.json').write_text(json.dumps(report, indent=2))
        (run / 'radio_inventory.json').write_text(json.dumps(inventory, indent=2))
    comparison = Path(spec['output_dir']) / f'comparison-{runs[0].name}.csv'
    compare_experiments(runs, comparison, cohort=spec.get('comparison_cohort', 'all'))
    print(f'Comparaison : {comparison}', flush=True)
PY
