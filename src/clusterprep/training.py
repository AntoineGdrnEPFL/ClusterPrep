"""Expériences supervisées reproductibles avec sélection sur validation seule."""

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re
import time
import uuid
import warnings
from .labels import labels_for, task_target, merge_cluster_labels
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from .dataset import ClusterDataset
from .io import CLASS_NAMES
from .metrics import (
    classification_metrics,
    multilabel_metrics,
    resolve_thresholds,
    select_multilabel_thresholds,
)
from .models import build_model, _list_models_with_scattering
from .splits import split_data_keys, split_distribution
from .processed import prepare_processed
from .progress import progress, message
from .training_core import (
    TrainingConfig,
    LEGACY_TRAINING_CONFIG,
    EarlyStopping,
    compute_class_weights,
    build_criterion,
    build_optimizer,
    build_scheduler,
    mixup_batch,
    move_batch_to_device,
    forward_model,
)
from .transfer import (
    TransferConfig,
    TransferPhase,
    load_pretrained_modules_from_state,
    read_checkpoint_state,
    configure_transfer_phase,
    set_transfer_train_mode,
    validate_transfer_plan,
    prepare_transfer_source,
)
from .modular_transfer import ModularTransferConfig, TransferError, initialize_transfer, parse_transfer
from .experiment_loader import resolve_transfer_splits, audit_transfer_splits


@dataclass(frozen=True)
class ExperimentConfig(TrainingConfig):
    name: str
    mode: str = "raw"
    model: str = "cnn"
    task: str = "de_nde"
    candidate_policy: str = "positive"
    thresholds: dict = field(default_factory=lambda: {"RR": 0.5, "RH": 0.5})
    threshold_selection: str = "fixed"
    threshold_min_precision: float = 0.8
    threshold_grid: list = field(
        default_factory=lambda: [i / 100 for i in range(5, 100, 5)]
    )
    transfer: TransferConfig | ModularTransferConfig | None = None
    training_phases: list[TransferPhase] = field(default_factory=list)
    seed: int = 42
    size: int = 128
    width: int = 16
    augment: bool = True
    cache: bool = True
    processed_dir: str | None = "processed"
    verbose: bool = True
    num_workers: int = 0
    device: str = "mps"
    monitor: str = "val_loss"
    min_delta: float = 0.0
    class_weight: str = "balanced"
    model_params: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "transfer", parse_transfer(self.transfer))
        if isinstance(self.transfer, TransferConfig) and self.task != "rr_rh":
            raise ValueError("Legacy transfer learning requires task='rr_rh'; use sources/mappings for other tasks")
        if not isinstance(self.training_phases, (list, tuple)):
            raise ValueError("training_phases must be a list")
        phases = [TransferPhase(**p) if isinstance(p, dict) else p for p in self.training_phases]
        if not all(isinstance(p, TransferPhase) for p in phases):
            raise ValueError("training_phases must contain phase objects")
        for phase in phases:
            phase.__post_init__()
        if len({p.name for p in phases}) != len(phases):
            raise ValueError("Duplicate training phase names")
        if phases and isinstance(self.transfer, TransferConfig):
            raise ValueError("Use either legacy transfer.phases or training_phases, not both")
        object.__setattr__(self, "training_phases", phases)
        resolve_thresholds(self.thresholds)
        if self.threshold_selection not in {"fixed", "f1", "recall_at_precision"}:
            raise ValueError("threshold_selection : fixed, f1 ou recall_at_precision")
        if self.task == "de_nde" and self.threshold_selection != "fixed":
            raise ValueError("La sélection de seuils est réservée à rr_rh")
        if (
            not np.isfinite(self.threshold_min_precision)
            or not 0 <= self.threshold_min_precision <= 1
        ):
            raise ValueError("threshold_min_precision doit être entre 0 et 1")
        if (
            not isinstance(self.threshold_grid, (list, tuple))
            or not self.threshold_grid
            or any(
                not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 < v < 1
                for v in self.threshold_grid
            )
        ):
            raise ValueError("threshold_grid : liste non vide de nombres entre 0 et 1")
        if not isinstance(self.model_params, dict) or not all(
            isinstance(k, str) for k in self.model_params
        ):
            raise ValueError("model_params doit être un dictionnaire à clés texte")
        try:
            json.dumps(self.model_params, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("model_params doit être sérialisable en JSON") from error
        if self.processed_dir is not None:
            object.__setattr__(self, "processed_dir", str(self.processed_dir))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name):
            raise ValueError("name doit être un identifiant sans chemin")
        if self.mode not in {"raw", "pair", "multimodal"}:
            raise ValueError("mode doit être raw, pair ou multimodal")
        for key in ("size", "width", "batch_size", "epochs", "patience"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} doit être un entier positif")
        if (
            self.size < 2
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("size >= 2 et num_workers >= 0 requis")
        for key in ("learning_rate", "weight_decay", "min_delta"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} doit être fini et positif ou nul")
        if self.learning_rate == 0:
            raise ValueError("learning_rate doit être strictement positif")
        if self.monitor not in {"val_loss", "val_macro_f1", "val_balanced_accuracy"}:
            raise ValueError(
                "monitor attendu : val_loss, val_macro_f1, val_balanced_accuracy"
            )
        if self.class_weight not in {"none", "balanced"}:
            raise ValueError("class_weight doit être none ou balanced")
        if self.task not in {"de_nde", "rr_rh"}:
            raise ValueError("task doit être de_nde ou rr_rh")
        if self.candidate_policy not in {"positive", "ignore"}:
            raise ValueError("candidate_policy doit être positive ou ignore")


def _training_phases(config):
    if isinstance(getattr(config, "transfer", None), TransferConfig):
        return config.transfer.phases
    return getattr(config, "training_phases", [])


class _MergeSingletonBatchSampler(BatchSampler):
    """Fusionne le dernier singleton avec le lot précédent, sans perdre de données."""

    def __iter__(self):
        previous = None
        for batch in super().__iter__():
            if previous is not None:
                if len(batch) == 1:
                    yield previous + batch
                    previous = None
                    continue
                yield previous
            previous = batch
        if previous is not None:
            yield previous

    def __len__(self):
        count = super().__len__()
        return count - int(count > 1 and len(self.sampler) % self.batch_size == 1)


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _validate_splits(
    info, splits, class_names, task="de_nde", candidate_policy="positive"
):
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError("Au moins deux classes distinctes requises")
    if len(splits) != 3:
        raise ValueError("Trois splits train/validation/test attendus")
    seen = set()
    for keys in splits:
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Splits non vides et sans clés dupliquées requis")
        clusters = set()
        for key in keys:
            task_target(info[key], task, candidate_policy, class_names)
            if info[key]["cluster_id"] in clusters:
                raise ValueError(
                    "Cluster dupliqué dans un split : dédupliquer les observations"
                )
            clusters.add(info[key]["cluster_id"])
        if seen & clusters:
            raise ValueError("Fuite de clusters entre les splits")
        seen.update(clusters)
    labels = {}
    for keys in splits:
        for key in keys:
            item = info[key]
            previous = labels.setdefault(item["cluster_id"], labels_for(item))
            if previous != labels_for(item):
                raise ValueError("Classes contradictoires pour un cluster")


def _inputs(batch, mode, model=None):
    names = ("raw",) if mode == "raw" else ("raw", "chandra", "chandra_mask")
    x = torch.cat([batch[name] for name in names], dim=1)
    if model is not None and getattr(
        model,
        "uses_precomputed_scattering",
        False,
    ):
        if "scattering" not in batch:
            raise ValueError("Le modèle requiert un scattering précalculé")
        return x, batch["scattering"]
    return x


def compute_task_counts(
    data_info,
    train_keys,
    task,
    candidate_policy="positive",
):
    targets = [task_target(data_info[k], task, candidate_policy)[0] for k in train_keys]
    if task == "de_nde":
        return [sum(t == i for t in targets) for i in range(2)]
    return np.asarray(targets).sum(axis=0).tolist()


def _epoch(
    model,
    loader,
    config,
    device,
    class_names,
    optimizer=None,
    weights=None,
    description=None,
    criterion=None,
    scheduler=None,
    input_adapter=None,
    frozen_modules=(),
):
    training = optimizer is not None
    model.train(training)
    if training and frozen_modules:
        set_transfer_train_mode(model, frozen_modules)
    labels, probabilities, predictions, masks = [], [], [], []
    loss_denominator = 0.0
    task = getattr(config, "task", "de_nde")
    total_loss, count = 0.0, 0
    criterion = criterion or (
        nn.BCEWithLogitsLoss()
        if task == "rr_rh"
        else nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)
    )
    with (
        torch.enable_grad() if training else torch.inference_mode(),
        progress(
            loader,
            description or ("Entraînement" if training else "Évaluation"),
            getattr(config, "verbose", False),
        ) as bar,
    ):
        for batch in bar:
            batch = move_batch_to_device(batch, device)
            target = batch["label"]
            if task == "rr_rh":
                observed = batch.get("label_mask", torch.ones_like(target))
                if (
                    observed.shape != target.shape
                    or not ((observed == 0) | (observed == 1)).all()
                ):
                    raise ValueError("Masque de labels binaire [B, 2] requis")
                target = torch.where(observed.bool(), target, torch.zeros_like(target))
            if training:
                optimizer.zero_grad(set_to_none=True)
            inputs = (
                input_adapter(batch)
                if input_adapter
                else _inputs(batch, getattr(config, "mode", "raw"), model)
            )
            mixed = (
                training
                and config.use_mixup
                and random.random() < config.mixup_probability
            )
            if mixed and getattr(model, "uses_precomputed_scattering", False):
                raise ValueError("MixUp incompatible avec scattering précalculé")
            if mixed:
                if task == "rr_rh":
                    packed = torch.stack(
                        (target, batch.get("label_mask", torch.ones_like(target))),
                        dim=1,
                    )
                    inputs, packed_a, packed_b, lam = mixup_batch(
                        inputs, packed, config.mixup_alpha
                    )
                    target_a, mask_a = packed_a.unbind(1)
                    target_b, mask_b = packed_b.unbind(1)
                else:
                    inputs, target_a, target_b, lam = mixup_batch(
                        inputs, target, config.mixup_alpha
                    )
            logits = forward_model(model, inputs)
            if (
                logits.shape != (len(target), len(class_names))
                or not torch.isfinite(logits).all()
            ):
                raise ValueError(
                    "Le modèle doit retourner des logits finis [B, nombre_classes]"
                )
            if getattr(config, "task", "de_nde") == "rr_rh":
                if target.shape != logits.shape:
                    raise ValueError(
                        f"Target multi-label invalide : "
                        f"{target.shape} != {logits.shape}"
                    )
                if target.dtype != torch.float32:
                    raise TypeError("Les cibles RR/RH doivent être float32")
                if (
                    not torch.isfinite(target).all()
                    or not ((target == 0) | (target == 1)).all()
                ):
                    raise ValueError("Targets binaires finis requis")
            # CE simple conservée séparément pour les comparaisons de cohortes.
            if getattr(config, "task", "de_nde") == "de_nde":
                losses = nn.functional.cross_entropy(
                    logits,
                    target,
                    reduction="none",
                )
                if mixed:
                    loss = lam * criterion(logits, target_a) + (1 - lam) * criterion(
                        logits, target_b
                    )
                else:
                    loss = criterion(logits, target)
            elif getattr(config, "task", "de_nde") == "rr_rh":
                mask = batch.get("label_mask", torch.ones_like(target))
                if mask.shape != target.shape or not ((mask == 0) | (mask == 1)).all():
                    raise ValueError("Masque de labels binaire [B, 2] requis")
                elements = nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
                losses = (elements * mask).sum(1) / mask.sum(1).clamp_min(1)

                def masked_loss(y, m):
                    values = nn.functional.binary_cross_entropy_with_logits(
                        logits, y, pos_weight=criterion.pos_weight, reduction="none"
                    )
                    return (values * m).sum(), m.sum()

                if mixed:
                    numerator_a, denominator_a = masked_loss(target_a, mask_a)
                    numerator_b, denominator_b = masked_loss(target_b, mask_b)
                    denominator = lam * denominator_a + (1 - lam) * denominator_b
                    loss = (
                        lam * numerator_a + (1 - lam) * numerator_b
                    ) / denominator.clamp_min(1)
                else:
                    numerator, denominator = masked_loss(target, mask)
                    loss = numerator / denominator.clamp_min(1)
                masks.extend(mask.cpu().tolist())
            else:
                raise ValueError(f"Tâche inconnue : {config.task}")
            if not torch.isfinite(loss):
                raise ValueError("Perte non finie")
            if training and (task != "rr_rh" or float(denominator) > 0):
                loss.backward()
                if config.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=config.gradient_clip_norm
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            if getattr(config, "task", "de_nde") == "de_nde":
                proba = logits.detach().softmax(dim=1).cpu().numpy()
            elif getattr(config, "task", "de_nde") == "rr_rh":
                proba = logits.detach().sigmoid().cpu().numpy()
            else:
                raise ValueError(f"Tâche inconnue : {config.task}")
            truth = target.cpu().numpy()
            if task == "rr_rh":
                batch_denominator = float(denominator)
            elif (
                isinstance(criterion, nn.CrossEntropyLoss)
                and criterion.weight is not None
            ):
                batch_denominator = float(criterion.weight[target].sum())
            else:
                batch_denominator = len(target)
            total_loss += loss.item() * batch_denominator
            loss_denominator += batch_denominator
            count += len(target)
            bar.set_postfix(
                loss=f"{total_loss / max(loss_denominator, 1):.4f}", refresh=False
            )
            labels.extend(truth.tolist())
            probabilities.extend(proba.tolist())
            if not training:
                sample_losses = losses.detach().cpu().tolist()
                for i in range(len(target)):
                    if getattr(config, "task", "de_nde") == "de_nde":
                        true_label = int(truth[i])
                        predicted_label = int(proba[i].argmax())
                    elif getattr(config, "task", "de_nde") == "rr_rh":
                        true_label = truth[i].astype(int).tolist()
                        predicted_label = (
                            (
                                proba[i]
                                >= np.array(
                                    list(
                                        resolve_thresholds(
                                            config.thresholds, class_names
                                        ).values()
                                    )
                                )
                            )
                            .astype(int)
                            .tolist()
                        )
                    else:
                        raise ValueError(f"Tâche inconnue : {config.task}")
                    predictions.append(
                        {
                            "key": batch.get(
                                "key", list(range(count - len(target), count))
                            )[i],
                            "cluster_id": batch.get("cluster_id", [None] * len(target))[
                                i
                            ],
                            "label": true_label,
                            "label_mask": (
                                mask[i].cpu().tolist() if task == "rr_rh" else None
                            ),
                            "prediction": predicted_label,
                            "has_chandra": bool(
                                batch.get("has_chandra", [False] * len(target))[i]
                            ),
                            "loss": sample_losses[i],
                            "probabilities": proba[i].tolist(),
                        }
                    )
    if loss_denominator == 0 and count:
        raise ValueError("Aucun label connu dans ce split")
    if count == 0:
        raise ValueError("Aucun exemple dans le DataLoader")
    if getattr(config, "task", "de_nde") == "de_nde":
        metrics = classification_metrics(
            labels,
            probabilities,
            class_names,
        )
    elif getattr(config, "task", "de_nde") == "rr_rh":
        metrics = multilabel_metrics(
            labels,
            probabilities,
            label_names=class_names,
            thresholds=getattr(config, "thresholds", None),
            mask=masks,
        )
    else:
        raise ValueError(f"Tâche inconnue : {config.task}")
    metrics["loss"] = total_loss / max(loss_denominator, 1)
    return metrics, predictions


def _record_history(run_dir, record):
    """Shared scalar CSV and full JSONL logs for normal and phased training."""
    with (run_dir / "history.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    flat = {k: record[k] for k in ("epoch", "seconds", "learning_rate")}
    for key in ("phase", "phase_epoch"):
        if key in record:
            flat[key] = record[key]
    for prefix, metrics in (("train", record["train"]), ("val", record["validation"])):
        flat.update(
            {
                f"{prefix}_{k}": v
                for k, v in metrics.items()
                if not isinstance(v, (dict, list))
            }
        )
        for label, scores in metrics.get("per_label", {}).items():
            flat.update(
                {
                    f"{prefix}_{label}_{k}": v
                    for k, v in scores.items()
                    if not isinstance(v, (dict, list))
                }
            )
    path = run_dir / "history.csv"
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat))
        if new:
            writer.writeheader()
        writer.writerow(flat)


class Trainer:
    """Moteur générique ; input_adapter(batch) sélectionne les entrées du modèle.

    L'adaptateur renvoie un tenseur, un tuple d'arguments ou un dict d'arguments
    nommés. Toutes ces entrées partagent le même MixUp. Les batches contiennent
    au minimum « label » ; les métadonnées de prédiction sont facultatives.
    """

    def __init__(
        self,
        model,
        config,
        class_names,
        class_counts,
        *,
        device="cpu",
        steps_per_epoch,
        input_adapter=None,
        n_train=None,
    ):
        self.model = model.to(device)
        self.config, self.class_names, self.device = config, tuple(class_names), device
        self.input_adapter = input_adapter
        self.criterion = build_criterion(config, class_counts, device, n_train=n_train)
        self.steps_per_epoch = steps_per_epoch
        self.frozen_modules = []
        self.optimizer = (
            None
            if _training_phases(config)
            else build_optimizer(model, config)
        )
        self.scheduler = (
            build_scheduler(self.optimizer, config, steps_per_epoch)
            if self.optimizer
            else None
        )
        monitor = getattr(config, "monitor", "val_loss")
        self.stopping = EarlyStopping(
            config.patience,
            getattr(config, "min_delta", 0.0),
            "min" if monitor == "val_loss" else "max",
        )
        self.best_epoch = 0
        self.stopped_early = False

    def train_one_epoch(self, loader, description=None):
        if self.optimizer is None:
            raise RuntimeError(
                "Transfer optimizer is not configured; call fit_transfer()"
            )
        return _epoch(
            self.model,
            loader,
            self.config,
            self.device,
            self.class_names,
            optimizer=self.optimizer,
            criterion=self.criterion,
            scheduler=self.scheduler,
            input_adapter=self.input_adapter,
            description=description,
            frozen_modules=self.frozen_modules,
        )

    def evaluate(self, loader, description=None):
        return _epoch(
            self.model,
            loader,
            self.config,
            self.device,
            self.class_names,
            criterion=self.criterion,
            input_adapter=self.input_adapter,
            description=description,
        )

    def fit(self, train_loader, val_loader, *, run_dir=None, phase=None):
        """Entraîne puis restaure le meilleur modèle, même sans arrêt anticipé.

        Créer un Trainer et un modèle neufs pour chaque expérience.
        """
        history = []
        if run_dir is not None:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, self.config.epochs + 1):
            started = time.perf_counter()
            if run_dir is None:
                train, _ = self.train_one_epoch(train_loader)
                validation, _ = self.evaluate(val_loader)
            else:
                train, _ = self.train_one_epoch(
                    train_loader, f"Époque {epoch}/{self.config.epochs} · train"
                )
                validation, _ = self.evaluate(
                    val_loader, f"Époque {epoch}/{self.config.epochs} · validation"
                )
            record = {
                "epoch": epoch,
                "seconds": time.perf_counter() - started,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "train": train,
                "validation": validation,
            }
            if phase is not None:
                record.update(
                    phase=phase,
                    phase_epoch=epoch,
                    learning_rates={
                        g["name"]: g["lr"] for g in self.optimizer.param_groups
                    },
                )
            history.append(record)
            if run_dir is not None:
                _record_history(run_dir, record)
            key = getattr(self.config, "monitor", "val_loss").removeprefix("val_")
            if validation[key] is None:
                raise ValueError(f"Métrique val_{key} non définie ; utiliser val_loss")
            improved, stop = self.stopping.update(validation[key], self.model)
            if improved:
                self.best_epoch = epoch
                if run_dir is not None:
                    torch.save(self._checkpoint(epoch, phase), run_dir / "best.pt")
            if getattr(self.config, "verbose", False):
                message(
                    f"Époque {epoch:03d}/{self.config.epochs} | loss train={train['loss']:.4f} "
                    f"val={validation['loss']:.4f} | F1 val={validation.get('macro_f1', 0.):.3f} | "
                    f"{'checkpoint enregistré' if improved else f'patience {self.stopping.bad_epochs}/{self.config.patience}'} "
                    f"| {record['seconds']:.1f} s",
                    True,
                )
            if stop and self.config.use_early_stopping:
                self.stopped_early = True
                message("Arrêt anticipé.", getattr(self.config, "verbose", False))
                break
        self.stopping.restore(self.model)
        return history

    def _checkpoint(self, epoch, phase=None):
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "config": asdict(self.config),
            "class_names": list(self.class_names),
            "epoch": epoch,
            "monitor": getattr(self.config, "monitor", "val_loss"),
            "score": self.stopping.best,
            "phase": phase,
            "preprocessing": {
                "size": getattr(self.config, "size", None),
                "mode": getattr(self.config, "mode", None),
                "normalization": {"percentile_lo": 30, "percentile_hi": 99, "alpha": 10.0},
            },
            "transfer_report": getattr(self, "initialization_report", None),
        }

    def fit_transfer(self, train_loader, val_loader, *, run_dir=None, phases=None):
        """Train phases through fit(), restoring each best before the next phase.

        Final selection compares phase bests on validation only. Test loaders
        are deliberately absent from this API.
        """
        original = self.config
        phase_plan = list(phases) if phases is not None else _training_phases(original)
        if not phase_plan:
            raise ValueError("fit_transfer requires training_phases or legacy config.transfer.phases")
        if isinstance(getattr(original, "transfer", None), TransferConfig):
            modules = validate_transfer_plan(self.model, replace(original.transfer, phases=phase_plan))
        else:
            from .transfer import validate_training_phases
            modules = validate_training_phases(self.model, phase_plan)
        if len(train_loader) < 1:
            raise ValueError("No training batches")
        if original.use_scheduler and any(
            original.warmup_fraction * p.epochs * len(train_loader) == 1
            for p in phase_plan
        ):
            raise ValueError(
                "OneCycleLR: warmup_fraction * total_steps must not equal 1 in any phase"
            )
        history, reports = [], []
        best_score = None
        selected_state = None
        selected_stopping = None
        selected_epoch = 0
        selected_phase = None
        any_stopped = False
        run_dir = Path(run_dir) if run_dir is not None else None
        monitor = getattr(original, "monitor", "val_loss")
        try:
            for phase in phase_plan:
                self.config = replace(
                    original,
                    epochs=phase.epochs,
                    patience=phase.patience,
                    weight_decay=phase.weight_decay,
                )
                self.optimizer, self.frozen_modules = configure_transfer_phase(
                    self.model, phase
                )
                self.scheduler = build_scheduler(
                    self.optimizer, self.config, len(train_loader)
                )
                self.stopping = EarlyStopping(
                    phase.patience,
                    getattr(original, "min_delta", 0.0),
                    "min" if monitor == "val_loss" else "max",
                )
                self.best_epoch, self.stopped_early = 0, False
                phase_dir = run_dir / "phases" / phase.name if run_dir else None
                offset = len(history)
                records = self.fit(
                    train_loader, val_loader, run_dir=phase_dir, phase=phase.name
                )
                phase_report = {
                    **asdict(phase),
                    "trainable_groups": list(phase.learning_rates),
                    "frozen_groups": sorted(set(modules) - set(phase.learning_rates)),
                    "epochs_ran": len(records),
                    "best_epoch": self.best_epoch,
                    "global_best_epoch": offset + self.best_epoch,
                    "monitor": monitor,
                    "monitor_best": self.stopping.best,
                    "stopped_early": self.stopped_early,
                    "validation": records[self.best_epoch - 1]["validation"],
                }
                reports.append(phase_report)
                if phase_dir:
                    _json(phase_dir / "metrics.json", phase_report)
                for record in records:
                    global_record = {**record, "epoch": offset + record["epoch"]}
                    history.append(global_record)
                    if run_dir:
                        _record_history(run_dir, global_record)
                score = self.stopping.best
                better = best_score is None or (
                    score < best_score if monitor == "val_loss" else score > best_score
                )
                if better:
                    best_score, selected_phase = score, phase.name
                    selected_epoch = offset + self.best_epoch
                    selected_stopping = self.stopping
                    selected_state = self.stopping.best_state_dict
                    if run_dir:
                        checkpoint = torch.load(
                            phase_dir / "best.pt", map_location="cpu", weights_only=True
                        )
                        checkpoint.update(
                            config=asdict(original),
                            phase_epoch=self.best_epoch,
                            epoch=selected_epoch,
                        )
                        torch.save(checkpoint, run_dir / "best.pt")
                any_stopped |= self.stopped_early
            self.model.load_state_dict(selected_state)
            self.stopping = selected_stopping
            self.best_epoch, self.stopped_early = selected_epoch, any_stopped
            self.transfer_report = {
                "phases": reports,
                "selected_phase": selected_phase,
                "selection": monitor,
                "epochs_ran": len(history),
            }
            return history
        finally:
            self.config = original

    def predict(self, loader):
        """Probabilités CPU, dans l'ordre du loader ; labels non nécessaires."""
        self.model.eval()
        predictions = []
        with torch.inference_mode():
            for batch in loader:
                batch = move_batch_to_device(batch, self.device)
                inputs = (
                    self.input_adapter(batch)
                    if self.input_adapter
                    else _inputs(batch, getattr(self.config, "mode", "raw"), self.model)
                )
                logits = forward_model(self.model, inputs)
                if getattr(self.config, "task", "de_nde") == "de_nde":
                    probabilities = logits.softmax(dim=1)
                elif getattr(self.config, "task", "de_nde") == "rr_rh":
                    probabilities = logits.sigmoid()
                else:
                    raise ValueError(f"Tâche inconnue : {self.config.task}")
                predictions.append(probabilities.cpu())
        if not predictions:
            raise ValueError("Aucun exemple dans le DataLoader")
        return torch.cat(predictions)


def _cohort(
    predictions,
    class_names,
    sources,
    task="de_nde",
    thresholds=None,
    candidate_policy=None,
):
    if not predictions:
        return {
            "cohort_id": None,
            "metrics": None,
        }
    ordered = sorted(
        predictions,
        key=lambda p: p["key"],
    )
    targets = [p["label"] for p in ordered]
    probabilities = [p["probabilities"] for p in ordered]
    if task == "de_nde":
        metrics = classification_metrics(
            targets,
            probabilities,
            class_names,
        )
    elif task == "rr_rh":
        metrics = multilabel_metrics(
            targets,
            probabilities,
            label_names=class_names,
            mask=[p.get("label_mask", [1, 1]) for p in ordered],
            thresholds=thresholds,
        )
    else:
        raise ValueError(f"Tâche inconnue : {task}")
    if task == "rr_rh":
        known = [sum(p.get("label_mask", [1, 1])) for p in ordered]
        metrics["loss"] = (
            float(np.average([p["loss"] for p in ordered], weights=known))
            if sum(known)
            else None
        )
    else:
        metrics["loss"] = float(np.mean([p["loss"] for p in ordered]))
    identity = {
        "task": task,
        "classes": list(class_names),
        **({"candidate_policy": candidate_policy} if task == "rr_rh" else {}),
        "samples": [
            (
                p["key"],
                p["cluster_id"],
                p["label"],
                p.get("label_mask"),
                sources[p["key"]],
            )
            for p in ordered
        ],
    }
    return {
        "cohort_id": _fingerprint(identity),
        "metrics": metrics,
    }


def _source_signature(item):
    signature = {}
    for name in ("raw_path", "chandra_path"):
        if item.get(name):
            path = Path(item[name]).resolve()
            stat = path.stat()
            signature[name] = {
                "path": str(path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        else:
            signature[name] = None
    return signature


def _write_predictions(
    path,
    predictions,
    class_names,
    task="de_nde",
):
    base_fields = [
        "key",
        "cluster_id",
        "has_chandra",
        "loss",
    ]
    probability_fields = [f"probability_{name}" for name in class_names]
    if task == "de_nde":
        fields = base_fields + ["label", "prediction"] + probability_fields
    elif task == "rr_rh":
        label_fields = [f"label_{name}" for name in class_names]
        prediction_fields = [f"prediction_{name}" for name in class_names]
        fields = (
            base_fields
            + label_fields
            + prediction_fields
            + probability_fields
            + [f"known_{name}" for name in class_names]
        )
    else:
        raise ValueError(f"Tâche inconnue : {task}")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
        )
        writer.writeheader()
        for prediction in predictions:
            row = {key: prediction[key] for key in base_fields}
            if task == "de_nde":
                row["label"] = prediction["label"]
                row["prediction"] = prediction["prediction"]
            elif task == "rr_rh":
                for i, name in enumerate(class_names):
                    row[f"label_{name}"] = prediction["label"][i]
                    row[f"known_{name}"] = prediction.get("label_mask", [1, 1])[i]
                    row[f"prediction_{name}"] = prediction["prediction"][i]
            for i, name in enumerate(class_names):
                row[f"probability_{name}"] = prediction["probabilities"][i]
            writer.writerow(row)


def _unique_cluster_splits(info, splits):
    """Choose one observation per cluster in automatically generated splits."""
    result = []
    for keys in splits:
        chosen = {}
        for key in keys:
            cid = info[key]["cluster_id"]
            if cid not in chosen or (
                info[key].get("chandra_path")
                and not info[chosen[cid]].get("chandra_path")
            ):
                chosen[cid] = key
        result.append(list(chosen.values()))
    return tuple(result)


def run_experiment(
    data_info,
    config,
    output_dir="results/experiments",
    *,
    class_names=CLASS_NAMES,
    splits=None,
    model_factory=None,
    input_adapter=None,
    split_metadata=None,
):
    """Entraîne un modèle neuf, restaure son meilleur état, puis évalue le test une fois.

    model_factory(config, num_classes) doit renvoyer un nn.Module prenant
    un tenseur [B, 1 ou 3, H, W] et retournant des logits [B, num_classes].
    input_adapter(batch) peut remplacer cette entrée par un tenseur, tuple ou dict.
    Chaque appel crée un répertoire unique et ne remplace aucune expérience.
    """
    data_info = merge_cluster_labels(data_info)
    class_names = ("RR", "RH") if config.task == "rr_rh" else tuple(class_names)
    if config.task == "de_nde" and set(class_names) != {"DE", "NDE"}:
        class_names = CLASS_NAMES
    transfer_splits_report = None
    if isinstance(config.transfer, TransferConfig) and config.transfer.reuse_splits:
        resolved, transfer_splits_report = resolve_transfer_splits(
            config.transfer.source_experiment,
            data_info,
            task=config.task,
            mode=config.mode,
            candidate_policy=config.candidate_policy,
            allow_subset=config.transfer.allow_subset,
            add_new_clusters=config.transfer.add_new_clusters,
            seed=config.transfer.seed,
        )
        if splits is not None and tuple(map(set, splits)) != tuple(map(set, resolved)):
            raise ValueError("Explicit splits conflict with source transfer splits")
        splits = resolved
    if splits is None:
        splits = split_data_keys(
            data_info,
            mode=(config.mode if config.task == "rr_rh" else "raw"),
            task=config.task,
            candidate_policy=config.candidate_policy,
            seed=config.seed,
        )
        splits = _unique_cluster_splits(data_info, splits)
    splits = tuple(list(keys) for keys in splits)
    _validate_splits(
        data_info, splits, class_names, config.task, config.candidate_policy
    )
    if isinstance(config.transfer, TransferConfig):
        audit = audit_transfer_splits(
            config.transfer.source_experiment, data_info, splits
        )
        transfer_splits_report = transfer_splits_report or {
            "split_policy": "explicit_or_generated"
        }
        transfer_splits_report["leakage_audit"] = audit
        for warning in audit["warnings"]:
            warnings.warn(warning, UserWarning)
    base_splits = splits
    if config.mode == "pair":
        splits = tuple(
            [k for k in keys if data_info[k].get("chandra_path")] for keys in splits
        )
    _validate_splits(
        data_info, splits, class_names, config.task, config.candidate_policy
    )
    class_counts = compute_task_counts(
        data_info,
        splits[0],
        config.task,
        config.candidate_policy,
    )
    if config.task == "de_nde" and tuple(class_names) == ("NDE", "DE"):
        class_counts.reverse()
    if min(class_counts) == 0:
        raise ValueError(
            "Chaque classe doit être présente dans l'entraînement après filtrage"
        )
    if getattr(config, "task", "de_nde") == "rr_rh":
        n_train = np.asarray(
            [
                task_target(data_info[k], config.task, config.candidate_policy)[1]
                for k in splits[0]
            ]
        ).sum(0)
        if any(count >= total for count, total in zip(class_counts, n_train)):
            raise ValueError(
                "Chaque label doit également avoir " "au moins un exemple négatif"
            )
    run_id = f"{config.name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _json(run_dir / "status.json", {"status": "running"})
    try:
        if config.model in _list_models_with_scattering() and config.use_mixup:
            print("⚠️  Modèle avec scattering : MixUp désactivé pour cette expérience.")
            config = replace(config, use_mixup=False)
        return _run(
            data_info,
            config,
            run_dir,
            class_names,
            splits,
            base_splits,
            class_counts,
            model_factory or build_model,
            input_adapter,
            split_metadata,
            transfer_splits_report,
        )
    except Exception as error:
        _json(run_dir / "status.json", {"status": "failed", "error": str(error)})
        raise


def _run(
    info,
    config,
    run_dir,
    class_names,
    splits,
    base_splits,
    class_counts,
    factory,
    input_adapter=None,
    split_metadata=None,
    transfer_splits_report=None,
):
    started = time.perf_counter()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    output_names = tuple(class_names)
    model = factory(
        config,
        len(output_names),
    ).to(device)
    transfer_report = None
    if isinstance(config.transfer, ModularTransferConfig) and config.transfer.enabled:
        try:
            transfer_report = initialize_transfer(model, config.transfer, factory)
        except TransferError as error:
            _json(run_dir / "transfer.json", error.report)
            raise
        _json(run_dir / "transfer.json", transfer_report)
        for entry in transfer_report["mappings"]:
            message(f"Transfer {entry['source']}:{entry['from']} -> {entry['to']}: "
                    f"{entry['status']} ({entry['reason']})", config.verbose)
    elif isinstance(config.transfer, TransferConfig):
        transfer_report = prepare_transfer_source(model, config, factory)
        transfer_report["split_provenance"] = transfer_splits_report
        _json(run_dir / "transfer.json", transfer_report)
        _json(run_dir / "transfer_splits_report.json", transfer_splits_report)

    needs_multiple = any(
        isinstance(layer, nn.modules.batchnorm._BatchNorm) for layer in model.modules()
    )
    train_count = len(splits[0]) * (ClusterDataset.N_VARIANTS if config.augment else 1)
    if needs_multiple and (config.batch_size < 2 or train_count < 2):
        raise ValueError(
            "Un modèle avec BatchNorm requiert batch_size >= 2 et au moins deux exemples train"
        )
    epoch_budget = (
        sum(p.epochs for p in _training_phases(config))
        if _training_phases(config)
        else config.epochs
    )
    message(
        f"\n{config.name} | mode={config.mode}, modèle={config.model}, device={device}, "
        f"seed={config.seed}\n"
        f"Train/validation/test : {'/'.join(str(len(k)) for k in splits)} amas ; "
        f"augmentation={'24 variantes' if config.augment else 'non'} ; "
        f"{epoch_budget} époques maximum, early stopping={config.monitor} "
        f"(patience={config.patience}).\nRésultats : {run_dir}",
        config.verbose,
    )
    preparation = None
    if config.processed_dir is not None:
        preparation = prepare_processed(
            info,
            [k for keys in splits for k in keys],
            config.mode,
            config.size,
            config.processed_dir,
            config.verbose,
            warning_log=run_dir / "fits_warnings.log",
        )
        _json(run_dir / "preprocessing.json", preparation)
    scattering_dir = None
    scattering_params = None
    if getattr(model, "uses_precomputed_scattering", False):
        if config.use_mixup:
            raise ValueError(
                "Le scattering précalculé n'est pas compatible "
                "exactement avec le MixUp d'entrée. "
                "Utiliser use_mixup=False pour DualSSN."
            )
        from .scattering import prepare_scattering

        scattering_params = model.scattering_config
        scattering_dir = Path(config.processed_dir or "processed") / "scattering"
        scattering_preparation = prepare_scattering(
            info,
            [k for keys in splits for k in keys],
            class_names,
            mode=config.mode,
            size=config.size,
            processed_dir=config.processed_dir,
            scattering_dir=scattering_dir,
            device=device,
            verbose=config.verbose,
            task=config.task,
            candidate_policy=config.candidate_policy,
            **scattering_params,
        )
        _json(
            run_dir / "scattering.json",
            scattering_preparation,
        )
    datasets = [
        ClusterDataset(
            info,
            keys,
            class_names,
            mode=config.mode,
            size=config.size,
            augment=config.augment and i == 0,
            cache=config.cache,
            processed_dir=config.processed_dir,
            scattering_dir=scattering_dir,
            scattering_params=scattering_params,
            task=config.task,
            candidate_policy=config.candidate_policy,
        )
        for i, keys in enumerate(splits)
    ]
    message(f"Exemples par split : {[len(ds) for ds in datasets]}", config.verbose)
    loaders = []
    for i, ds in enumerate(datasets):
        generator = torch.Generator().manual_seed(config.seed + i)
        batching = {
            "batch_size": config.batch_size,
            "shuffle": i == 0,
            "drop_last": i == 0 and config.drop_last_train,
        }
        if i == 0 and needs_multiple and not config.drop_last_train:
            batching = {
                "batch_sampler": _MergeSingletonBatchSampler(
                    RandomSampler(ds, generator=generator),
                    config.batch_size,
                    drop_last=False,
                )
            }
        loaders.append(
            DataLoader(
                ds,
                **batching,
                num_workers=config.num_workers,
                worker_init_fn=_seed_worker,
                generator=generator,
                persistent_workers=config.num_workers > 0,
            )
        )
    trainer = Trainer(
        model,
        config,
        output_names,
        class_counts,
        device=device,
        steps_per_epoch=len(loaders[0]),
        input_adapter=input_adapter,
        n_train=(
            np.asarray(
                [
                    task_target(info[k], config.task, config.candidate_policy)[1]
                    for k in splits[0]
                ]
            ).sum(0)
            if config.task == "rr_rh"
            else len(splits[0])
        ),
    )
    metadata = {
        "schema_version": 2,
        "config": asdict(config),
        "class_names": list(class_names),
        "train_class_counts": dict(zip(class_names, class_counts)),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_repr": repr(model),
        "scattering_config": getattr(model, "scattering_config", None),
        "library_source_sha256": _fingerprint(
            {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))}
        ),
        "parameters": sum(p.numel() for p in model.parameters()),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "astropy",
                "reproject",
                "scikit-learn",
            )
        },
        "normalization": {"percentile_lo": 30, "percentile_hi": 99, "alpha": 10.0},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _json(run_dir / "config.json", metadata)
    split_manifest = {
        name: keys for name, keys in zip(("train", "validation", "test"), splits)
    }
    if split_metadata is not None:
        split_manifest["split_config"] = split_metadata
    split_manifest["base_splits"] = dict(
        zip(("train", "validation", "test"), base_splits)
    )
    split_manifest["samples"] = {
        k: {
            **info[k],
            "labels": sorted(labels_for(info[k])),
            "raw_path": str(Path(info[k]["raw_path"]).resolve()),
            "chandra_path": (
                str(Path(info[k]["chandra_path"]).resolve())
                if info[k].get("chandra_path")
                else None
            ),
        }
        for keys in base_splits
        for k in keys
    }
    sources = {k: _source_signature(info[k]) for keys in base_splits for k in keys}
    split_manifest["source_signatures"] = sources
    _json(run_dir / "splits.json", split_manifest)
    _json(
        run_dir / "split_distribution.json",
        split_distribution(info, splits, config.task, config.candidate_policy),
    )
    trainer.initialization_report = transfer_report
    if _training_phases(config):
        history = trainer.fit_transfer(loaders[0], loaders[1], run_dir=run_dir)
        _json(run_dir / "training_phases.json", trainer.transfer_report)
        if isinstance(config.transfer, TransferConfig):
            transfer_report.update(trainer.transfer_report)
            _json(run_dir / "transfer.json", transfer_report)
    else:
        history = trainer.fit(loaders[0], loaders[1], run_dir=run_dir)
    stopping = trainer.stopping
    best_epoch, stopped = trainer.best_epoch, trainer.stopped_early
    val_metrics, val_predictions = trainer.evaluate(
        loaders[1], description="Validation finale"
    )
    threshold_selection = None
    if config.task == "rr_rh":
        threshold_selection = select_multilabel_thresholds(
            [p["label"] for p in val_predictions],
            [p["probabilities"] for p in val_predictions],
            target_mask=[p["label_mask"] for p in val_predictions],
            method=config.threshold_selection,
            grid=config.threshold_grid,
            min_precision=config.threshold_min_precision,
            fallback=config.thresholds,
        )
        chosen = threshold_selection["thresholds"]
        config = replace(config, thresholds=chosen)
        trainer.config = config
        validation_loss = val_metrics["loss"]
        val_metrics = multilabel_metrics(
            [p["label"] for p in val_predictions],
            [p["probabilities"] for p in val_predictions],
            target_mask=[p["label_mask"] for p in val_predictions],
            thresholds=chosen,
        )
        val_metrics["loss"] = validation_loss
        for prediction in val_predictions:
            prediction["prediction"] = (
                (
                    np.asarray(prediction["probabilities"])
                    >= np.array(list(chosen.values()))
                )
                .astype(int)
                .tolist()
            )
        _json(run_dir / "thresholds.json", threshold_selection)
        metadata["evaluation"] = threshold_selection
        _json(run_dir / "config.json", metadata)
        checkpoint = torch.load(
            run_dir / "best.pt", map_location="cpu", weights_only=True
        )
        checkpoint["evaluation"] = threshold_selection
        torch.save(checkpoint, run_dir / "best.pt")
    test_metrics, test_predictions = trainer.evaluate(
        loaders[2], description="Test final"
    )
    _write_predictions(
        run_dir / "validation_predictions.csv",
        val_predictions,
        output_names,
        task=config.task,
    )
    _write_predictions(
        run_dir / "test_predictions.csv",
        test_predictions,
        output_names,
        task=config.task,
    )
    results = {
        "schema_version": 2,
        "run_id": run_dir.name,
        "config": asdict(config),
        "class_names": list(class_names),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "stopped_early": stopped,
        "monitor_best": stopping.best,
        "seconds": time.perf_counter() - started,
        "preprocessing": preparation,
        "split_counts": dict(zip(("train", "validation", "test"), map(len, splits))),
        "validation": val_metrics,
        "test": test_metrics,
        **(
            {
                "threshold_selection": threshold_selection,
                "annotation_convention": "unannotated_negative",
            }
            if config.task == "rr_rh"
            else {}
        ),
        "test_cohorts": {
            "all": _cohort(
                test_predictions,
                output_names,
                sources,
                task=config.task,
                thresholds=config.thresholds,
                candidate_policy=config.candidate_policy,
            ),
            "paired": _cohort(
                [p for p in test_predictions if p["has_chandra"]],
                output_names,
                sources,
                task=config.task,
                thresholds=config.thresholds,
                candidate_policy=config.candidate_policy,
            ),
            "raw_only": _cohort(
                [p for p in test_predictions if not p["has_chandra"]],
                output_names,
                sources,
                task=config.task,
                thresholds=config.thresholds,
                candidate_policy=config.candidate_policy,
            ),
        },
    }
    if isinstance(config.transfer, TransferConfig):
        audit = transfer_splits_report["leakage_audit"]
        historical = set().union(*map(set, audit["source_cluster_ids"].values()))
        cohorts = {
            "historical": [
                p for p in test_predictions if str(p["cluster_id"]) in historical
            ],
            "new_candidates": [
                p
                for p in test_predictions
                if str(p["cluster_id"]) not in historical
                and labels_for(info[p["key"]]) & {"cRR", "cRH"}
            ],
            "diffuse_emission": [
                p
                for p in test_predictions
                if labels_for(info[p["key"]]) & {"DE", "RR", "RH", "cRR", "cRH"}
            ],
        }
        for name, predictions in cohorts.items():
            results["test_cohorts"][name] = _cohort(
                predictions,
                output_names,
                sources,
                task=config.task,
                thresholds=config.thresholds,
                candidate_policy=config.candidate_policy,
            )
    if transfer_report is not None:
        results["transfer"] = transfer_report
    _json(run_dir / "metrics.json", results)
    _json(run_dir / "status.json", {"status": "completed"})
    message(
        f"Terminé : test loss={test_metrics['loss']:.4f}, "
        f"F1 macro={test_metrics['macro_f1']:.3f}, accuracy={test_metrics['accuracy']} "
        f"| {results['seconds']:.1f} s\nFichiers : {run_dir}",
        config.verbose,
    )
    return run_dir


def run_experiments(
    data_info,
    configs,
    output_dir="results/experiments",
    *,
    class_names=CLASS_NAMES,
    splits=None,
    split_seed=42,
    test_size=0.2,
    val_size=0.2,
    n_folds=None,
    fold_index=0,
    model_factory=None,
    input_adapter=None,
):
    """Suite séquentielle partageant les splits, indépendamment des graines modèle."""
    configs = list(configs)
    if not configs:
        raise ValueError("Au moins une configuration d'expérience requise")
    supplied_splits = splits
    split_cache = {}
    runs = []
    for index, config in enumerate(configs, 1):
        if isinstance(config.transfer, TransferConfig) and config.transfer.reuse_splits:
            message(f"\nExpérience {index}/{len(configs)}", config.verbose)
            runs.append(
                run_experiment(
                    data_info,
                    config,
                    output_dir,
                    class_names=class_names,
                    splits=supplied_splits,
                    model_factory=model_factory,
                    input_adapter=input_adapter,
                )
            )
            continue
        split_mode = (
            "pair" if config.task == "rr_rh" and config.mode == "pair" else "raw"
        )
        split_key = (config.task, config.candidate_policy, split_mode)
        if supplied_splits is None and split_key not in split_cache:
            split_cache[split_key] = split_data_keys(
                data_info,
                mode=split_mode,
                seed=split_seed,
                task=config.task,
                candidate_policy=config.candidate_policy,
                test_size=test_size,
                val_size=val_size,
                n_folds=n_folds,
                fold_index=fold_index,
            )
            split_cache[split_key] = _unique_cluster_splits(
                data_info, split_cache[split_key]
            )
        splits = (
            supplied_splits if supplied_splits is not None else split_cache[split_key]
        )
        message(f"\nExpérience {index}/{len(configs)}", config.verbose)
        runs.append(
            run_experiment(
                data_info,
                config,
                output_dir,
                class_names=class_names,
                splits=splits,
                model_factory=model_factory,
                input_adapter=input_adapter,
                split_metadata=(
                    {
                        "split_seed": split_seed,
                        "n_folds": n_folds,
                        "fold_index": fold_index,
                        "val_size": val_size,
                        "test_size": test_size,
                    }
                    if supplied_splits is None
                    else None
                ),
            )
        )
    return runs


def compare_experiments(run_dirs, output_path=None, cohort="paired"):
    """Table de comparaison ; refuse les cohortes ou classes de test différentes."""
    if cohort not in {"all", "paired", "raw_only"}:
        raise ValueError("Cohorte attendue : all, paired ou raw_only")
    rows, cohort_id = [], None
    run_dirs = list(run_dirs)
    tasks = [
        json.loads((Path(d) / "metrics.json").read_text())
        .get("config", {})
        .get("task", "de_nde")
        for d in run_dirs
    ]
    canonical_cohort_id = None
    if tasks and all(task == "rr_rh" for task in tasks):
        from .multilabel_analysis import load_run, assert_comparable

        canonical_cohort_id = assert_comparable(
            [load_run(d) for d in run_dirs], cohort=cohort
        )
    for directory in run_dirs:
        result = json.loads((Path(directory) / "metrics.json").read_text())
        selected = result["test_cohorts"][cohort]
        if canonical_cohort_id is not None:
            selected = {**selected, "cohort_id": canonical_cohort_id}
        if selected["cohort_id"] is None:
            raise ValueError(f"Cohorte vide pour {directory}")
        if cohort_id is not None and cohort_id != selected["cohort_id"]:
            raise ValueError(
                "Cohortes de test différentes : comparaison directe refusée"
            )
        cohort_id = selected["cohort_id"]
        row = {
            "run_id": result["run_id"],
            "name": result["config"]["name"],
            "mode": result["config"]["mode"],
            "model": result["config"]["model"],
            "seed": result["config"]["seed"],
            "best_epoch": result["best_epoch"],
            "seconds": result["seconds"],
            "cohort": cohort,
            "cohort_id": cohort_id,
        }
        row.update(
            {
                k: v
                for k, v in selected["metrics"].items()
                if not isinstance(v, (dict, list))
            }
        )
        if result["config"].get("task", "de_nde") == "rr_rh":
            per_label = selected["metrics"]["per_label"]
            for label_name, label_metrics in per_label.items():
                for metric_name, value in label_metrics.items():
                    if not isinstance(value, (dict, list)):
                        row[f"{label_name}_{metric_name}"] = value
        if result["config"].get("task", "de_nde") == "rr_rh":
            row["candidate_policy"] = result["config"].get(
                "candidate_policy", "unknown"
            )
            row["threshold_selection"] = result.get("threshold_selection", {}).get(
                "method", "fixed"
            )
            for name, cutoff in resolve_thresholds(
                result["config"].get("thresholds")
            ).items():
                row[f"threshold_{name}"] = cutoff
        rows.append(row)
    if not rows:
        raise ValueError("Au moins une expérience requise")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(dict.fromkeys(key for row in rows for key in row)),
            )
            writer.writeheader()
            writer.writerows(rows)
    return rows
