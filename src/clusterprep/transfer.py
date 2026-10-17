"""Utilities for explicit, reproducible transfer learning in clusterprep.

Every participating model must implement ``transfer_modules()`` returning a
mapping from logical role (e.g. radio, xray, fusion, head) to an nn.Module.
Those modules must form a disjoint, exhaustive partition of trainable model
parameters (parameterless modules are allowed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
import math
import re
import json
import hashlib

import torch
from torch import nn


@dataclass
class TransferPhase:
    name: str
    epochs: int
    learning_rates: dict[str, float]
    weight_decay: float = 1e-3
    patience: int = 10

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name):
            raise ValueError("A transfer phase needs a name without path components")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (self.epochs, self.patience)):
            raise ValueError("epochs and patience must be >= 1")
        if not isinstance(self.learning_rates, dict) or not self.learning_rates:
            raise ValueError("learning_rates cannot be empty")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and >= 0")
        for name, lr in self.learning_rates.items():
            if not isinstance(name, str) or not name or isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
                raise ValueError(f"Invalid learning rate for {name}: {lr}")


@dataclass
class TransferConfig:
    source_experiment: str
    reuse_splits: bool = True
    add_new_clusters: bool = True
    allow_subset: bool = False
    seed: int = 42
    module_prefixes: dict[str, str] = field(default_factory=dict)
    phases: list[TransferPhase] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.source_experiment, (str, Path)) or not str(self.source_experiment).strip():
            raise ValueError("source_experiment is required")
        self.source_experiment = str(Path(self.source_experiment).expanduser())
        for name in ("reuse_splits", "add_new_clusters", "allow_subset"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if not self.reuse_splits and (self.add_new_clusters or self.allow_subset):
            raise ValueError("add_new_clusters/allow_subset require reuse_splits")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not isinstance(self.module_prefixes, dict) or not self.module_prefixes:
            raise ValueError("module_prefixes cannot be empty")
        if "head" in self.module_prefixes:
            raise ValueError("Do not transfer the DE/NDE head to the RR/RH task")
        for name, prefix in self.module_prefixes.items():
            if not isinstance(name, str) or not name or not isinstance(prefix, str) or not prefix or prefix.endswith("."):
                raise ValueError("module_prefixes must map group names to nonempty prefixes without trailing dots")
        if not isinstance(self.phases, (list, tuple)) or not self.phases:
            raise ValueError("At least one transfer phase is required")
        self.phases = [TransferPhase(**p) if isinstance(p, dict) else p for p in self.phases]
        if not all(isinstance(p, TransferPhase) for p in self.phases):
            raise TypeError("phases must contain TransferPhase instances or dictionaries")
        for phase in self.phases:
            phase.__post_init__()
        if len({p.name for p in self.phases}) != len(self.phases):
            raise ValueError("Duplicate transfer phase names")


def validate_transfer_plan(model, config):
    """Validate all phases before any optimizer step or checkpoint mutation."""
    modules = validate_transfer_modules(model)
    if "head" not in modules:
        raise ValueError("A transfer model must identify its classification head")
    for phase in config.phases:
        phase.__post_init__()
        unknown = set(phase.learning_rates) - set(modules)
        if unknown:
            raise ValueError(f"Unknown trainable groups in {phase.name}: {sorted(unknown)}")
        if not any(list(modules[name].parameters()) for name in phase.learning_rates):
            raise ValueError(f"No trainable parameters in phase {phase.name}")
    unknown = set(config.module_prefixes) - set(modules)
    if unknown:
        raise ValueError(f"Unknown target modules: {sorted(unknown)}")
    # Newly initialized branches must learn immediately, including multimodal ones.
    fresh = {name for name, module in modules.items() if list(module.parameters())} - set(config.module_prefixes)
    missing = fresh - set(config.phases[0].learning_rates)
    if missing:
        raise ValueError(f"First phase must train newly initialized groups: {sorted(missing)}")
    return modules


def _logical_modules(model: nn.Module) -> dict[str, nn.Module]:
    if not callable(getattr(model, "transfer_modules", None)):
        raise TypeError(f"{type(model).__name__} does not define transfer_modules()")
    modules = model.transfer_modules()
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("transfer_modules() must return a nonempty mapping")
    for name, module in modules.items():
        if not isinstance(name, str) or not name or not isinstance(module, nn.Module):
            raise TypeError("transfer_modules() must map nonempty strings to nn.Module")
    return dict(modules)


def validate_transfer_modules(model: nn.Module) -> dict[str, nn.Module]:
    """Require all model parameters to belong to exactly one logical group."""
    modules = _logical_modules(model)
    registered = {id(module) for module in model.modules()}
    if any(id(module) not in registered for module in modules.values()):
        raise ValueError("Transfer groups must be registered modules of the model")
    owners = {}
    for name, module in modules.items():
        for parameter in module.parameters():
            pid = id(parameter)
            if pid in owners:
                raise ValueError(
                    f"Overlapping transfer groups: {owners[pid]!r} and {name!r}"
                )
            owners[pid] = name
    model_ids = {id(p) for p in model.parameters()}
    if set(owners) - model_ids:
        raise ValueError("Transfer groups contain parameters outside the model")
    missing = model_ids - set(owners)
    if missing:
        raise ValueError(
            f"{len(missing)} model parameters are not in transfer_modules()"
        )
    return modules


def read_checkpoint_state(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Read weights from a plain state_dict or a supported checkpoint wrapper."""
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")
    for field in ("model_state_dict", "model_state", "state_dict", "model"):
        if field in checkpoint and isinstance(checkpoint[field], Mapping):
            checkpoint = checkpoint[field]
            break
    if not checkpoint or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor)
        for k, v in checkpoint.items()
    ):
        raise ValueError("Checkpoint contains no recognized tensor state_dict")
    return dict(checkpoint)


def _legacy_image_state(state):
    """Only the documented pre-ImageEncoder layout is migrated; no fuzzy matching."""
    if "encoder.0.weight" not in state or "to_latent.0.weight" not in state or "FC_classifier.weight" not in state:
        return state, {}
    names = {}
    for key in state:
        if key.startswith("encoder."):
            names[key] = "encoder." + key
        elif key.startswith("to_latent."):
            names[key] = "encoder." + key
    converted = {names.get(key, key): tensor for key, tensor in state.items()}
    if len(converted) != len(state):
        raise ValueError("Ambiguous legacy ImageCNN checkpoint")
    return converted, names


def prepare_transfer_source(target_model, config, factory):
    """Check source metadata and encoder semantics, then strictly import weights.

    Source construction preserves the caller's RNG so target training remains
    reproducible. No optimizer, classifier or scheduler state is imported.
    """
    from .training import ExperimentConfig

    transfer = config.transfer
    modules = validate_transfer_plan(target_model, transfer)
    source_dir = Path(transfer.source_experiment).expanduser().resolve()
    metadata = json.loads((source_dir / "config.json").read_text())
    raw = metadata.get("config", metadata)
    source_config = ExperimentConfig(**raw)
    if source_config.task != "de_nde" or set(metadata.get("class_names", ["DE", "NDE"])) != {"DE", "NDE"}:
        raise ValueError("Source experiment must be a DE/NDE experiment")
    if source_config.size != config.size:
        raise ValueError(f"Incompatible image size: source={source_config.size}, target={config.size}")
    normalization = {"percentile_lo": 30, "percentile_hi": 99, "alpha": 10.0}
    warnings = []
    if "normalization" not in metadata:
        warnings.append("Source normalization metadata absent; historical clusterprep defaults assumed")
    elif metadata["normalization"] != normalization:
        raise ValueError("Incompatible source/target normalization")
    with torch.random.fork_rng(devices=[]):
        source_model = factory(source_config, 2)
    source_modules = validate_transfer_modules(source_model)
    named = dict(source_model.named_modules())
    head = source_modules.get("head")
    head_ids = {id(m) for m in head.modules()} if head is not None else set()
    for target, prefix in transfer.module_prefixes.items():
        if prefix not in named:
            raise ValueError(f"Unknown source prefix {prefix!r}; available prefixes: {sorted(named)}")
        selected = named[prefix]
        if any(id(m) in head_ids for m in selected.modules()):
            raise ValueError("Do not transfer the DE/NDE head to the RR/RH task")
        if type(selected) is not type(modules[target]) or repr(selected) != repr(modules[target]):
            raise ValueError(f"Incompatible encoder architecture/channels: {prefix} -> {target}")
        if hasattr(selected, "scattering"):
            for name in ("J", "L", "max_order"):
                if getattr(selected.scattering, name, None) != getattr(modules[target].scattering, name, None):
                    raise ValueError(f"Incompatible scattering parameter: {name}")
    checkpoint_path = source_dir / "best.pt"
    report = load_pretrained_modules_from_state(target_model, checkpoint_path, transfer.module_prefixes)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if any(item.get("legacy_key_mapping") for item in report.values()):
        warnings.append("Legacy ImageCNN encoder/to_latent keys explicitly migrated to ImageEncoder")
    return {
        "source_experiment": str(source_dir), "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": digest.hexdigest(),
        "source_architecture": source_config.model, "target_architecture": config.model,
        "source_config": raw, "module_prefixes": transfer.module_prefixes,
        "loaded_modules": report, "compatibility_warnings": warnings,
        "seed": config.seed, "split_seed": transfer.seed,
        "candidate_policy": config.candidate_policy,
    }


def _check_compatible_states(source_state, target_state, description):
    source_keys, target_keys = set(source_state), set(target_state)
    if source_keys != target_keys:
        raise ValueError(
            f"Incompatible {description}: missing in source={sorted(target_keys-source_keys)}, "
            f"unexpected in source={sorted(source_keys-target_keys)}"
        )
    mismatches = [
        name
        for name in source_state
        if source_state[name].shape != target_state[name].shape
        or source_state[name].dtype != target_state[name].dtype
    ]
    if mismatches:
        raise ValueError(f"Shape/dtype mismatch for {description}: {mismatches}")


def transfer_model_modules(
    source_model: nn.Module,
    target_model: nn.Module,
    module_mapping: dict[str, str],
) -> dict:
    """Copy selected logical modules, including buffers, with strict compatibility."""
    source_modules = validate_transfer_modules(source_model)
    target_modules = validate_transfer_modules(target_model)
    if not module_mapping:
        raise ValueError("module_mapping cannot be empty")
    if len(set(module_mapping.values())) != len(module_mapping):
        raise ValueError("Each target logical module can be mapped only once")
    if "head" in module_mapping or "head" in module_mapping.values():
        raise ValueError("Do not transfer the DE/NDE head to the RR/RH task")

    # Validate EVERY mapping before modifying the target model.
    checked = []
    for source_name, target_name in module_mapping.items():
        if source_name not in source_modules:
            raise ValueError(f"Unknown source transfer module: {source_name}")
        if target_name not in target_modules:
            raise ValueError(f"Unknown target transfer module: {target_name}")
        src_state = source_modules[source_name].state_dict()
        dst_state = target_modules[target_name].state_dict()
        if not src_state:
            raise ValueError(f"No state to transfer: {source_name}")
        _check_compatible_states(
            src_state, dst_state, f"{source_name} -> {target_name}"
        )
        checked.append((source_name, target_name, src_state))

    report = {}
    for source_name, target_name, src_state in checked:
        target_module = target_modules[target_name]
        target_module.load_state_dict(src_state, strict=True)
        report[target_name] = {
            "source_module": source_name,
            "num_tensors": len(src_state),
            "num_parameters": sum(p.numel() for p in target_module.parameters()),
        }
    return report


def load_pretrained_modules_from_state(
    target_model: nn.Module,
    checkpoint_path: str | Path,
    module_prefixes: dict[str, str],
) -> dict:
    """Load logical target modules from an existing checkpoint's named prefixes.

    module_prefixes maps target logical name -> prefix in the source state_dict,
    e.g. {"radio": "encoder"}. Unlike model-to-model transfer, this does not
    need to instantiate the source model, but requires explicit source prefixes.
    """
    modules = validate_transfer_modules(target_model)
    source_state, legacy_names = _legacy_image_state(read_checkpoint_state(checkpoint_path))
    if not module_prefixes:
        raise ValueError("module_prefixes cannot be empty")
    if "head" in module_prefixes:
        raise ValueError("Do not transfer the DE/NDE head to the RR/RH task")

    checked = []
    for target_name, prefix in module_prefixes.items():
        if target_name not in modules:
            raise ValueError(f"Unknown target module: {target_name}")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("Source prefix must be a nonempty string")
        if any(part in {"head", "classifier", "FC_input", "FC_hidden", "FC_classifier"} for part in prefix.split(".")):
            raise ValueError("Do not transfer the DE/NDE head to the RR/RH task")
        head = f"{prefix}."
        part = {
            k[len(head) :]: v for k, v in source_state.items() if k.startswith(head)
        }
        if not part:
            available = sorted({k.rsplit(".", 1)[0] for k in source_state})
            raise ValueError(f"No checkpoint state found under {head!r}; available prefixes: {available}")
        _check_compatible_states(part, modules[target_name].state_dict(), target_name)
        checked.append((target_name, prefix, part))

    report = {}
    for target_name, prefix, state in checked:
        modules[target_name].load_state_dict(state, strict=True)
        parameters = dict(modules[target_name].named_parameters())
        report[target_name] = {
            "source_prefix": prefix, "num_tensors": len(state),
            "parameters": [k for k in state if k in parameters],
            "buffers": [k for k in state if k not in parameters],
            "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in state.items()},
            "legacy_key_mapping": {old: new for old, new in legacy_names.items() if new.startswith(prefix + ".")},
        }
    return report


def configure_transfer_phase(
    model: nn.Module,
    phase: TransferPhase | Mapping[str, float],
    weight_decay: float | None = None,
) -> tuple[torch.optim.Optimizer, list[nn.Module]]:
    """Freeze unspecified groups; create a fresh optimizer for selected groups.

    Returns (optimizer, frozen_modules). Call set_transfer_train_mode() AFTER
    every model.train() invocation, including those inside the train loop.
    """
    modules = validate_transfer_modules(model)
    if isinstance(phase, TransferPhase):
        learning_rates = phase.learning_rates
        decay = phase.weight_decay if weight_decay is None else weight_decay
    else:
        learning_rates = dict(phase)
        decay = 1e-3 if weight_decay is None else weight_decay

    unknown = set(learning_rates) - set(modules)
    if unknown:
        raise ValueError(f"Unknown trainable groups: {sorted(unknown)}")
    if not learning_rates:
        raise ValueError("No trainable groups specified")
    if not math.isfinite(decay) or decay < 0:
        raise ValueError("weight_decay must be finite and >= 0")
    for name, lr in learning_rates.items():
        if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"Invalid learning rate for {name}: {lr}")

    # Only change requires_grad after validating inputs.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    param_groups = []
    for name, lr in learning_rates.items():
        params = list(modules[name].parameters())
        for parameter in params:
            parameter.requires_grad_(True)
        if params:
            param_groups.append({"params": params, "lr": lr, "name": name})
    if not param_groups:
        raise ValueError("No trainable parameters in selected groups")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=decay)
    frozen_modules = [
        module for name, module in modules.items() if name not in learning_rates
    ]
    return optimizer, frozen_modules


def set_transfer_train_mode(model: nn.Module, frozen_modules: list[nn.Module]) -> None:
    """Set training mode while keeping frozen groups' BatchNorm/buffers fixed."""
    model.train()
    for module in frozen_modules:
        module.eval()
