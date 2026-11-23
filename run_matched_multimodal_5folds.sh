
#!/usr/bin/env bash
set -euo pipefail

for product in RAW T50kpcSUB; do
    for fold in {0..4}; do
        config="clusterprep/configs/matched_historical/${product}/fold_${fold}.json"

        echo "=== Training ${product} — Fold ${fold} ==="
        python -m clusterprep "$config"
    done
done

echo "All training runs completed!"
