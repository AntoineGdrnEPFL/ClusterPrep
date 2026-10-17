from clusterprep.io import get_data_info_dict
from clusterprep.experiment_loader import resolve_transfer_splits
import json

data_info = get_data_info_dict(
    "/Users/antoinegdrn/PH_Project_I/scratch/data/PSZ2/classified",
    class_names=("DE", "NDE", "RR", "RH", "cRR", "cRH", "cDE"),
)

splits, report = resolve_transfer_splits(
    source_experiment=(
        "/Users/antoinegdrn/PH_Project_I/results/clusterprep/dual_encoder_cnn/dual_encoder_multimodal-20261003T211540-10302900"
    ),
    data_info=data_info,
    task="rr_rh",
    mode="raw",
    candidate_policy="positive",
)

train_keys, val_keys, test_keys = splits

print("Train :", len(train_keys))
print("Validation :", len(val_keys))
print("Test :", len(test_keys))
print("Excluded :", report["total_excluded"])
json_report = json.dumps(report, indent=2)
print("Report:")
print(json_report)
