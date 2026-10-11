import numpy as np

from clusterprep.metrics import multilabel_metrics

targets = np.array(
    [
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ]
)

probabilities = np.array(
    [
        [0.90, 0.10],
        [0.15, 0.85],
        [0.80, 0.75],
        [0.20, 0.30],
    ]
)

metrics = multilabel_metrics(
    targets,
    probabilities,
    label_names=("RR", "RH"),
)

print("Macro F1 :", metrics["macro_f1"])
print("ROC-AUC :", metrics["roc_auc_macro"])
print("RR :", metrics["per_label"]["RR"])
print("RH :", metrics["per_label"]["RH"])
