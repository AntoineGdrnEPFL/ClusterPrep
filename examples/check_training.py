"""Test court des trois modes sur 20 amas réels (pas un benchmark scientifique)."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from clusterprep import (CLASS_NAMES, ExperimentConfig, get_data_info_dict,
                         run_experiments, compare_experiments, plot_history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/clusterprep_training_check"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    info = get_data_info_dict(args.data)
    splits = ([], [], [])
    # Chaque split comporte les deux classes et les deux disponibilités Chandra.
    for label in CLASS_NAMES:
        for paired in (True, False):
            keys = [k for k, v in info.items() if v["class"] == label
                    and bool(v["chandra_path"]) == paired][:5]
            if len(keys) < 5:
                raise ValueError(f"Au moins cinq amas {label}, paired={paired}, requis")
            splits[0].extend(keys[:3])
            splits[1].append(keys[3])
            splits[2].append(keys[4])
    base = ExperimentConfig("smoke_raw", epochs=3, patience=1, min_delta=100.,
                            size=32, width=4, batch_size=4, augment=False)
    configs = [base, replace(base, name="smoke_pair", mode="pair"),
               replace(base, name="smoke_multimodal", mode="multimodal", model="fusion")]
    runs = run_experiments(info, configs, args.output, splits=splits)
    comparison = args.output / f"comparison-{runs[0].name}.csv"
    compare_experiments(runs, comparison)
    for run in runs:
        fig = plot_history(run, run / "learning_curves.png")
        plt.close(fig)
        results = json.loads((run / "metrics.json").read_text())
        assert results["best_epoch"] == 1
        assert results["epochs_ran"] == 2
        assert results["stopped_early"]
    print(json.dumps({"runs": list(map(str, runs)), "comparison": str(comparison)}, indent=2))


if __name__ == "__main__":
    main()
