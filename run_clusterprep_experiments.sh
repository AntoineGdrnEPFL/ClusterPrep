#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    PYTHON_BIN=python3
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Interpréteur Python introuvable : $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/clusterprep/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_DIR="$PROJECT_ROOT/clusterprep/configs/cv"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/clusterprep-cv.XXXXXX")"
trap 'rm -rf "$TEMP_DIR"' EXIT

configs=(
  dual_ssn.json
  dual_encoder_ssn.json
)

for config_name in "${configs[@]}"; do
  source_config="$CONFIG_DIR/$config_name"
  [[ -f "$source_config" ]] || {
    echo "Configuration introuvable : $source_config" >&2
    exit 1
  }

  model_name="${config_name%.json}"
  for fold in 0 1 2 3 4; do
    fold_config="$TEMP_DIR/${model_name}-fold-${fold}.json"
    sed "s/\\\"fold_index\\\": 0/\\\"fold_index\\\": $fold/" \
      "$source_config" > "$fold_config"

    echo "=== $model_name, fold $fold/4 ==="
    "$PYTHON_BIN" -m clusterprep "$fold_config"
  done
done
