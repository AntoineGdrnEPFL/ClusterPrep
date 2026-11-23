#!/usr/bin/env bash
# Ten sequential DualEncoderCNN runs using the saved Planck splits.
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
    config="clusterprep/configs/de_nde_raw_chandra_planck_splits_5folds/fold_${fold}.json"
    echo "DE/NDE, fold ${fold}: RAW then RAW+CHANDRA (saved Planck splits)"
    if [[ "$dry_run" == true ]]; then
        "$PYTHON_BIN" -m clusterprep "$config" --dry-run
    else
        "$PYTHON_BIN" -m clusterprep "$config"
    fi
done
