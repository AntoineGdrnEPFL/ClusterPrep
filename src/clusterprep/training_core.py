"""Protocole d'entraînement indépendant des architectures et des modalités."""
import copy
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    use_class_weights: bool = True
    label_smoothing: float = 0.1
    use_scheduler: bool = True
    scheduler: str = "onecycle"
    max_lr_factor: float = 3.0
    warmup_fraction: float = 0.3
    use_mixup: bool = True
    mixup_probability: float = 0.5
    mixup_alpha: float = 0.4
    gradient_clip_norm: float = 1.0  # 0 désactive le clipping
    use_early_stopping: bool = True
    patience: int = 50
    drop_last_train: bool = True

    def __post_init__(self):
        for name in ("epochs", "batch_size", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} doit être un entier positif")
        for name in ("learning_rate", "weight_decay", "label_smoothing", "max_lr_factor",
                     "warmup_fraction", "mixup_probability", "mixup_alpha", "gradient_clip_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} doit être fini et positif ou nul")
        if self.learning_rate <= 0 or self.max_lr_factor <= 0 or self.mixup_alpha <= 0:
            raise ValueError("learning_rate, max_lr_factor et mixup_alpha doivent être positifs")
        if not 0 < self.warmup_fraction < 1:
            raise ValueError("warmup_fraction doit être entre 0 et 1 exclusivement")
        if self.label_smoothing > 1 or self.mixup_probability > 1:
            raise ValueError("label_smoothing et mixup_probability doivent être entre 0 et 1")
        if self.scheduler != "onecycle":
            raise ValueError("scheduler doit être onecycle")
        for name in ("use_class_weights", "use_scheduler", "use_mixup", "use_early_stopping", "drop_last_train"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} doit être booléen")


LEGACY_TRAINING_CONFIG = TrainingConfig()


def compute_class_weights(class_counts, device="cpu"):
    counts = torch.as_tensor(class_counts, dtype=torch.float32, device=device)
    if counts.ndim != 1 or len(counts) < 2 or not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Effectifs finis strictement positifs requis pour chaque classe")
    return counts.sum() / counts


def build_criterion(config, class_counts, device="cpu"):
    enabled = config.use_class_weights and getattr(config, "class_weight", "balanced") != "none"
    weights = compute_class_weights(class_counts, device) if enabled else None
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)


def build_optimizer(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def build_scheduler(optimizer, config, steps_per_epoch):
    if steps_per_epoch < 1:
        raise ValueError("Aucun batch train : réduire batch_size ou désactiver drop_last_train")
    if not config.use_scheduler:
        return None
    if config.warmup_fraction * config.epochs * steps_per_epoch == 1:
        raise ValueError("OneCycleLR : warmup_fraction * total_steps ne doit pas être égal à 1")
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate * config.max_lr_factor,
        steps_per_epoch=steps_per_epoch, epochs=config.epochs,
        pct_start=config.warmup_fraction, anneal_strategy="cos")


def _map_tensors(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return {k: _map_tensors(v, function) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_map_tensors(v, function) for v in value)
    if isinstance(value, list):
        return [_map_tensors(v, function) for v in value]
    return value


def move_batch_to_device(batch, device):
    return _map_tensors(batch, lambda tensor: tensor.to(device))


def mixup_batch(inputs, target, alpha=0.4):
    """Mélange tous les tenseurs d'entrée avec une permutation et un lambda uniques.

    inputs contient uniquement les modalités du modèle (tenseur, tuple ou dict),
    pas les labels ni les identifiants. Les masques deviennent des masques souples.
    """
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha doit être fini et strictement positif")
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(len(target), device=target.device)
    def mix(tensor):
        if tensor.ndim == 0 or tensor.shape[0] != len(target):
            raise ValueError("Chaque modalité doit avoir une dimension batch commune")
        return lam * tensor + (1 - lam) * tensor[permutation.to(tensor.device)]
    return _map_tensors(inputs, mix), target, target[permutation], lam


def forward_model(model, inputs):
    """Tenseur -> model(x), tuple -> model(*x), dict -> model(**x)."""
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.0, mode="min"):
        if patience < 1 or min_delta < 0 or not math.isfinite(min_delta) or mode not in {"min", "max"}:
            raise ValueError("Paramètres d'early stopping invalides")
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best = None
        self.bad_epochs = 0
        self.best_state_dict = None
        self.should_stop = False

    def step(self, value, model=None):
        if not math.isfinite(value):
            raise ValueError("Métrique de validation non finie")
        improved = (self.best is None or
                    (value < self.best - self.min_delta if self.mode == "min"
                     else value > self.best + self.min_delta))
        if improved:
            self.best, self.bad_epochs = float(value), 0
            if model is not None:
                self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.bad_epochs += 1
        self.should_stop = self.bad_epochs >= self.patience
        return improved, self.should_stop

    update = step

    def restore(self, model):
        if self.best_state_dict is None:
            raise RuntimeError("Aucun meilleur état du modèle enregistré")
        model.load_state_dict(self.best_state_dict)
