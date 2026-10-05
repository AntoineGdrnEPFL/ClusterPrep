"""Test d'intégration reproductible : chargement, lots PyTorch et figures."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from clusterprep import (CLASS_NAMES, ClusterDataset, get_data_info_dict,
                         split_data_keys, plot_sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/clusterprep"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=4,
                        help="Échantillons par mode, 0 = tous")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit doit être positif ou nul")
    info = get_data_info_dict(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"data": str(args.data.resolve()), "classes": dict(Counter(
        item["class"] for item in info.values())), "modes": {}}
    for mode in ("raw", "pair"):
        train, val, test = split_data_keys(info, mode=mode)
        keys = train + val + test
        if args.limit:
            keys = keys[:args.limit]
        dataset = ClusterDataset(info, keys, CLASS_NAMES, mode=mode, size=args.size)
        start = time.perf_counter()
        count, coverage, batch_shapes = 0, [], []
        for batch in DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0):
            for name in ("raw", "chandra", "chandra_mask"):
                if name in batch:
                    assert torch.isfinite(batch[name]).all(), name
                    assert batch[name].shape[1:] == (1, args.size, args.size)
            count += len(batch["label"])
            batch_shapes.append(list(batch["raw"].shape))
            if mode == "pair":
                coverage.extend(batch["chandra_mask"].mean((1, 2, 3)).tolist())
        seconds = time.perf_counter() - start
        for i in range(min(3, len(dataset))):
            fig = plot_sample(dataset[i], args.output / f"{mode}_{i}.png")
            plt.close(fig)
        report["modes"][mode] = {"split_counts": [len(train), len(val), len(test)],
            "loaded": count, "seconds": seconds, "batch_shapes": batch_shapes,
            "chandra_coverage": coverage}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
