#!/usr/bin/env bash
# Five DualEncoderCNN RAW+Planck RR/RH runs using the saved 236-cluster RAW splits.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
usage() {
    echo "Usage: $0 [--dry-run] [0 1 2 3 4]"
    echo 'Historical preprocessing; final test after every training run.'
    echo '--dry-run validates configurations without training.'
}
dry_run=false
folds=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=true ;;
        -h|--help) usage; exit 0 ;;
        0|1|2|3|4)
            for selected in "${folds[@]:-}"; do
                if [[ "$selected" == "$arg" ]]; then
                    echo "Duplicate fold: $arg" >&2; exit 2
                fi
            done
            folds+=("$arg") ;;
        *) echo "Invalid argument: $arg" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ ${#folds[@]} -eq 0 ]]; then folds=(0 1 2 3 4); fi
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/clusterprep/src${PYTHONPATH:+:$PYTHONPATH}"
for fold in "${folds[@]}"; do
    config="clusterprep/configs/rr_rh_raw_planck_raw_splits_5folds/fold_${fold}.json"
    echo "RR/RH, fold ${fold}: RAW+Planck (236 clusters; original RAW splits)"
    if [[ "$dry_run" == true ]]; then
        "$PYTHON_BIN" -m clusterprep "$config" --dry-run
    else
        "$PYTHON_BIN" -m clusterprep "$config"
    fi
done
