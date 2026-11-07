#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
  echo "Usage : $0 [--dry-run] [0 1 2 3 4]"
  echo "Lance les modèles fusion et dual_encoder_cnn en rr_binary et rh_binary sur les folds choisis, d’abord gaussian_pixel puis beam_correlated."
  echo "Ordre : tous les folds gaussiens, puis tous les folds corrélés. Baseline non lancée."
  echo "Sans indice : les cinq folds (40 entraînements). PYTHON_BIN permet de choisir Python."
}

DRY_RUN=false
folds=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    -h|--help) usage; exit 0 ;;
    0|1|2|3|4)
      for selected in "${folds[@]:-}"; do
        if [[ "$selected" == "$arg" ]]; then
          echo "Fold indiqué plusieurs fois : $arg" >&2
          exit 2
        fi
      done
      folds+=("$arg")
      ;;
    *) echo "Argument invalide : $arg" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ ${#folds[@]} -eq 0 ]]; then
  folds=(0 1 2 3 4)
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    PYTHON_BIN=python
  fi
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Interpréteur Python introuvable : $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/clusterprep/src${PYTHONPATH:+:$PYTHONPATH}"
CONFIG_DIR="$PROJECT_ROOT/clusterprep/configs/radio_noise"

# Vérifier tous les fichiers avant de démarrer le premier entraînement.
methods=(gaussian_pixel beam_correlated)
for method in "${methods[@]}"; do
  for fold in "${folds[@]}"; do
    source_config="$CONFIG_DIR/$method/fold_${fold}.json"
    if [[ ! -f "$source_config" ]]; then
      echo "Configuration introuvable : $source_config" >&2
      exit 1
    fi
  done
done

for method in "${methods[@]}"; do
  for fold in "${folds[@]}"; do
    source_config="$CONFIG_DIR/$method/fold_${fold}.json"
    echo "=== $method : fold $fold/4, fusion et dual_encoder_cnn, RR et RH binaires ==="
    if [[ "$DRY_RUN" == true ]]; then
      printf '%q ' "$PYTHON_BIN" -m clusterprep "$source_config"
      printf '\n'
    else
      "$PYTHON_BIN" -m clusterprep "$source_config"
    fi
  done
done
