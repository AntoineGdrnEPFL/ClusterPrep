"""Préparation radio/Chandra pour l'entraînement de modèles."""
from .io import CLASS_NAMES, get_data_info_dict, get_pairs, load_celestial_fits
from .preprocessing import block_average, centered_raw_window, normalize, preprocess
from .dataset import ClusterDataset
from .splits import split_data_keys
from .visualization import plot_sample, plot_comparison, plot_history
from .models import (SimpleCNN, FusionCNN, ImageCNN, DualSSN, DualEncoderSSN,
                     build_model, register_model, list_models)
from .metrics import classification_metrics
from .training import (ExperimentConfig, EarlyStopping, run_experiment,
                       run_experiments, compare_experiments)

__all__ = ["CLASS_NAMES", "get_data_info_dict", "get_pairs", "load_celestial_fits",
           "block_average", "centered_raw_window", "normalize", "preprocess",
           "ClusterDataset", "split_data_keys", "plot_sample", "plot_comparison",
           "SimpleCNN", "FusionCNN", "build_model", "classification_metrics",
           "ExperimentConfig", "EarlyStopping", "run_experiment", "run_experiments",
           "compare_experiments", "plot_history"]

from .processed import load_processed, prepare_processed, processed_path, processed_paths
__all__ += ["load_processed", "prepare_processed", "processed_path", "processed_paths"]
__all__ += ["ImageCNN", "DualSSN", "DualEncoderSSN", "register_model", "list_models"]
from .config_wizard import create_config
__all__ += ["create_config"]

from .training import Trainer
from .training_core import (TrainingConfig, LEGACY_TRAINING_CONFIG, compute_class_weights,
    build_criterion, build_optimizer, build_scheduler, mixup_batch,
    move_batch_to_device, forward_model)
__all__ += ["Trainer", "TrainingConfig", "LEGACY_TRAINING_CONFIG", "compute_class_weights",
            "build_criterion", "build_optimizer", "build_scheduler", "mixup_batch",
            "move_batch_to_device", "forward_model"]
