"""Compare scratch/head-only/fine-tuning through the public training API."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np

from clusterprep import (ExperimentConfig, get_data_info_dict, resolve_transfer_splits,
                         run_experiment, compare_experiments, plot_history)
from clusterprep.multilabel_analysis import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", default=Path("clusterprep/examples/transfer_image_cnn.json"))
    parser.add_argument("--head-epochs", type=int)
    parser.add_argument("--finetune-epochs", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    base = ExperimentConfig(**spec["experiments"][0])
    if base.model != "image_cnn" or base.mode != "raw" or base.transfer is None or len(base.transfer.phases) != 2:
        parser.error("Pilot requires radio ImageCNN and exactly two transfer phases")
    phases = list(base.transfer.phases)
    if args.head_epochs is not None:
        phases[0] = replace(phases[0], epochs=args.head_epochs)
    if args.finetune_epochs is not None:
        phases[1] = replace(phases[1], epochs=args.finetune_epochs)
    base = replace(base, transfer=replace(base.transfer, phases=phases),
                   device=args.device or base.device, augment=False if args.no_augment else base.augment)
    info = get_data_info_dict(spec["data_dir"], spec.get("class_names"))
    t = base.transfer
    splits, provenance = resolve_transfer_splits(t.source_experiment, info, task=base.task,
        mode=base.mode, candidate_policy=base.candidate_policy, allow_subset=t.allow_subset,
        add_new_clusters=t.add_new_clusters, seed=t.seed)
    output = Path(spec["output_dir"]) / f"pilot-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "split_provenance.json").write_text(json.dumps(provenance, indent=2))
    experiments = [
        replace(base, name="scratch", transfer=None, epochs=sum(p.epochs for p in phases),
                learning_rate=1e-4, weight_decay=phases[-1].weight_decay, patience=phases[-1].patience),
        replace(base, name="classifier_only", transfer=replace(t, phases=phases[:1])),
        replace(base, name="full_finetuning"),
    ]
    (output / "pilot_config.json").write_text(json.dumps({**spec, "experiments": [asdict(c) for c in experiments]}, indent=2))
    historical_ids = {cid for s in provenance["splits"].values() for cid in s["historical_cluster_ids"]}
    runs, analysis = [], []
    for c in experiments:
        run = run_experiment(info, c, output, splits=splits,
                             split_metadata={"source_experiment": t.source_experiment, "seed": t.seed})
        runs.append(run)
        result = load_run(run)
        predictions = result.predictions["test"]
        selections = {
            "all": np.ones(len(predictions.keys), dtype=bool),
            "historical": np.array([str(cid) in historical_ids for cid in predictions.cluster_ids]),
            "new_candidates": np.array([str(cid) not in historical_ids for cid in predictions.cluster_ids]),
            "diffuse_emission": np.array([bool(info[str(k)]["labels"] & {"DE", "RR", "RH", "cRR", "cRH"}) for k in predictions.keys]),
        }
        analysis.append({
            "name": c.name, "run": str(run),
            "selected_phase": result.metrics.get("transfer", {}).get("selected_phase"),
            "cohorts": {name: predictions.subset(mask).scores(result.thresholds) if mask.any() else None
                        for name, mask in selections.items()},
        })
        import matplotlib.pyplot as plt
        plt.close(plot_history(run, run / "history.png"))
    compare_experiments(runs, output / "comparison.csv", cohort="all")
    (output / "pilot_metrics.json").write_text(json.dumps(analysis, indent=2, allow_nan=False))
    print(json.dumps({"output": str(output), "runs": list(map(str, runs))}, indent=2))


if __name__ == "__main__":
    main()
