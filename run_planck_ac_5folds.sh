#!/usr/bin/env bash
# A (RAW+Planck), then C (RAW+CHANDRA+Planck), with a final test per fold.
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$workspace"
geometry="${1:-matched_resolution}"
if [[ $# -gt 0 ]]; then shift; fi
case "$geometry" in
    historical|matched_resolution) ;;
    *) echo 'Usage: bash clusterprep/run_planck_ac_5folds.sh [historical|matched_resolution] [--dry-run]' >&2; exit 2 ;;
esac
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != '--dry-run' ) ]]; then
    echo 'Only --dry-run is accepted after the preprocessing name.' >&2
    exit 2
fi
for experiment in A C; do
    for fold in {0..4}; do
        config="clusterprep/configs/planck/${geometry}/${experiment}/fold_${fold}.json"
        echo "Planck ${experiment}, ${geometry}, fold ${fold}: training + final test"
        "$workspace/.venv/bin/python" -m clusterprep "$config" "$@"
    done
done
