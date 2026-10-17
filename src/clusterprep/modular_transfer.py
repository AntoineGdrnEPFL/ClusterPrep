"""Task-independent, explicit weight initialization (never training resume).

Custom modules opt in with a JSON-serializable ``transfer_signature()`` that
versions their forward semantics and includes all construction-time options.
Children and tensor schemas are always checked independently.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import copy
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import warnings

import numpy as np
import torch
from torch import nn

from .transfer import _check_compatible_states, _legacy_image_state, checkpoint_state

_LOG = logging.getLogger(__name__)
_RULES = {
    "compatibility": {"strict"},
    "on_missing": {"error", "warn", "skip"},
    "on_incompatible": {"error", "warn", "skip"},
    "on_conflict": {"error"},
}
_DEFAULTS = dict(compatibility="strict", on_missing="warn", on_incompatible="skip", on_conflict="error")


def _rules(value, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    for key, item in value.items():
        if key not in _RULES or not isinstance(item, str) or item not in _RULES[key]:
            raise ValueError(f"Invalid {where}.{key}: {item!r}; supported rules: {_RULES}")


def _path(value, where):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{where} must be a nonempty filesystem path")


@dataclass
class ModularTransferConfig:
    enabled: bool = False
    defaults: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    mappings: list = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("transfer.enabled must be boolean")
        if not self.enabled:
            return  # Disabled plans never inspect paths or sources.
        _rules(self.defaults, "transfer.defaults")
        if not isinstance(self.sources, dict) or not self.sources:
            raise ValueError("transfer.sources must be a nonempty object")
        for name, source in self.sources.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(source, dict):
                raise ValueError("transfer.sources must map nonempty names to objects")
            if set(source) - {"checkpoint", "config_path"} or "checkpoint" not in source:
                raise ValueError(f"Source {name!r} requires checkpoint and optionally config_path")
            for key, value in source.items():
                _path(value, f"transfer.sources.{name}.{key}")
        if not isinstance(self.mappings, list) or not self.mappings:
            raise ValueError("transfer.mappings must be a nonempty list")
        for index, mapping in enumerate(self.mappings):
            where = f"transfer.mappings[{index}]"
            if not isinstance(mapping, dict) or not {"source", "from", "to"} <= mapping.keys():
                raise ValueError(f"{where} requires source, from and to")
            if not isinstance(mapping["source"], str) or mapping["source"] not in self.sources:
                raise ValueError(f"{where}.source references an unknown source")
            for key in ("from", "to"):
                if not isinstance(mapping[key], str) or not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*", mapping[key]):
                    raise ValueError(f"{where}.{key} must be a dotted module path or logical name")
            _rules({k: v for k, v in mapping.items() if k not in {"source", "from", "to"}}, where)


def parse_transfer(value):
    """Keep legacy JSON unmodified; never interpret a mixed schema silently."""
    from .transfer import TransferConfig
    if isinstance(value, dict):
        cls = ModularTransferConfig if set(value) & {"enabled", "defaults", "sources", "mappings"} or not value else TransferConfig
        try:
            value = cls(**value)
        except TypeError as error:
            raise ValueError(f"Invalid transfer configuration: {error}") from error
    if value is not None:
        if not isinstance(value, (TransferConfig, ModularTransferConfig)):
            raise TypeError("transfer must be a transfer configuration or dictionary")
        value.__post_init__()
    return value


def unwrap_model(model):
    while isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    return model


def resolve_module(model, name):
    """Logical groups first, then aliases and registered PyTorch module paths."""
    model = unwrap_model(model)
    groups = model.transfer_modules() if callable(getattr(model, "transfer_modules", None)) else {}
    if not isinstance(groups, Mapping) or any(not isinstance(m, nn.Module) for m in groups.values()):
        raise ValueError("transfer_modules() must map names to modules")
    registered = {id(m) for m in model.modules()}
    if any(id(m) not in registered for m in groups.values()):
        raise ValueError("Transfer groups must be registered modules")
    first, *rest = name.split(".")
    aliases = {"radio_encoder": "radio", "xray_encoder": "xray", "classifier": "head"}
    if first in groups:
        module = groups[first]
    else:
        # Actual paths take priority over convenience aliases.
        module = model._modules.get(first)
        if module is None:
            module = groups.get(aliases.get(first))
    if module is None:
        raise KeyError(f"Module {name!r} absent from {type(model).__name__}")
    try:
        return module.get_submodule(".".join(rest)) if rest else module
    except AttributeError as error:
        raise KeyError(f"Module {name!r} absent from {type(model).__name__}") from error


def _json_value(value):
    # Reject arbitrary repr strings, functions and tensors: they cannot prove semantics.
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _json_value(v) for k, v in value.items()}
    raise ValueError(f"Unverifiable architecture option of type {type(value).__name__}")


def architecture_signature(module):
    cls = type(module)
    if any(getattr(module, attr, {}) for attr in (
        "_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
        "_state_dict_hooks", "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks",
    )) or "forward" in vars(module):
        raise ValueError("Hooks or instance forward overrides require explicit architectural validation")
    provider = vars(module).get("transfer_signature") or cls.__dict__.get("transfer_signature")
    if provider is not None:
        options = _json_value(module.transfer_signature())
    elif cls.__module__.startswith("torch.nn.modules."):
        options = {k: _json_value(v) for k, v in vars(module).items()
                   if not k.startswith("_") and k != "training"}
    else:
        raise ValueError(f"Custom module {cls.__module__}.{cls.__qualname__} requires transfer_signature()")
    json.dumps(options, allow_nan=False)
    return {
        "type": f"{cls.__module__}.{cls.__qualname__}",
        "options": options,
        "children": [(name, architecture_signature(child) if child is not None else None)
                     for name, child in module._modules.items()],
        "parameters": [(name, p is not None) for name, p in module._parameters.items()],
        "buffers": [(name, b is not None, name in module._non_persistent_buffers_set)
                    for name, b in module._buffers.items()],
    }


def _tensors(module):
    return dict(list(module.named_parameters(remove_duplicate=False)) + list(module.named_buffers(remove_duplicate=False)))


def _sharing(module):
    groups = {}
    for name, tensor in _tensors(module).items():
        groups.setdefault(_storage(tensor), []).append(name)
    return sorted(sorted(names) for names in groups.values())


def check_module_compatibility(source, target):
    if architecture_signature(source) != architecture_signature(target):
        raise ValueError("Incompatible architecture: module types, order or forward options differ")
    _check_compatible_states(_tensors(source), _tensors(target), "parameters/buffers")
    _check_compatible_states(source.state_dict(), target.state_dict(), "state_dict")
    if _sharing(source) != _sharing(target):
        raise ValueError("Incompatible shared parameter/buffer topology")


def _storage(tensor):
    return (str(tensor.device), tensor.untyped_storage()._cdata)


def _footprint(module):
    return {id(m) for m in module.modules()}, {_storage(t) for t in _tensors(module).values()}


class TransferError(ValueError):
    """Runtime failure carrying a JSON report, including unattempted mappings."""
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _load_source(spec, factory, checkpoint_cache):
    from .models import output_names_for_task, task_for_output_names
    from .training import ExperimentConfig
    path = Path(spec["checkpoint"]).expanduser().resolve()
    if path not in checkpoint_cache:
        checkpoint_cache[path] = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint = checkpoint_cache[path]
    metadata_origin = "checkpoint"
    if "config_path" in spec:
        config_path = Path(spec["config_path"]).expanduser().resolve()
        metadata = json.loads(config_path.read_text())
        metadata_origin = str(config_path)
    elif isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("config"), dict):
        metadata = checkpoint
    elif (path.parent / "config.json").is_file():
        config_path = path.parent / "config.json"
        metadata = json.loads(config_path.read_text())
        metadata_origin = str(config_path)
    else:
        raise ValueError("Missing reconstruction metadata; provide source.config_path (source experiment config)")
    if not isinstance(metadata, dict):
        raise ValueError("Source metadata must be a JSON object")
    raw = metadata.get("config", metadata)
    if not isinstance(raw, dict) or not {"model", "mode", "size"} <= raw.keys():
        raise ValueError("Source reconstruction config requires model, mode and size; provide config_path")
    raw = dict(raw)
    migrations = []
    if "task" not in raw:
        raw["task"] = task_for_output_names(metadata.get("class_names"))
        migrations.append({"field": "task", "value": raw["task"], "derived_from": "class_names"})
        _LOG.info("Legacy source %s: task=%s recovered from class_names", path, raw["task"])
    # Reconstruct architecture only: never execute a source's transfer plan/phases.
    source_config = ExperimentConfig(**{**raw, "transfer": None, "training_phases": []})
    names = metadata.get("class_names") or output_names_for_task(source_config.task)
    if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError("Source class_names must be a nonempty list of strings")
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            model = unwrap_model(factory(source_config, len(names))).cpu()
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    state = checkpoint_state(checkpoint)
    if state and all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    state, migrated = _legacy_image_state(state)
    _check_compatible_states(state, model.state_dict(), "source checkpoint reconstruction")
    model.load_state_dict(state, strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return model, {
        "checkpoint": str(path), "sha256": digest.hexdigest(), "config": raw,
        "metadata_origin": metadata_origin, "metadata_migrations": migrations,
        "task": source_config.task,
        "class_names": list(names), "legacy_key_mapping": migrated,
        "preprocessing": metadata.get("preprocessing", {
            "size": source_config.size, "mode": source_config.mode,
            "normalization": metadata.get("normalization"),
        }),
    }


def initialize_transfer(target, config, factory=None, *, resume=False):
    """Preflight the entire plan, then copy modules atomically, preserving RNG.

    Sources are cached by checkpoint/config path, including failed loads. Fatal
    preflight errors do not modify the target. Optimizers are never loaded.
    """
    from .models import build_model
    if not isinstance(config, ModularTransferConfig):
        raise TypeError("initialize_transfer requires ModularTransferConfig")
    config.__post_init__()
    report = {"enabled": config.enabled, "configuration": asdict(config), "sources": {}, "mappings": []}
    if not config.enabled:
        return report
    if resume:
        raise TransferError("resume and transfer initialization are mutually exclusive", report)
    factory = factory or build_model
    entries = report["mappings"]
    for mapping in config.mappings:
        entries.append({"source": mapping["source"], "checkpoint": str(Path(config.sources[mapping["source"]]["checkpoint"]).expanduser().resolve()),
                        "from": mapping["from"], "to": mapping["to"], "status": "skipped",
                        "reason": "Not applied: plan has not passed preflight", "num_parameters": 0,
                        "rules": {**_DEFAULTS, **config.defaults, **{k: v for k, v in mapping.items() if k in _RULES}}})

    def reject(entry, policy, reason):
        entry.update(status="error" if policy == "error" else "skipped", reason=reason)
        _LOG.warning("Transfer %s -> %s: %s", entry["from"], entry["to"], reason)
        if policy == "error":
            raise TransferError(reason, report)
        if policy == "warn":
            warnings.warn(reason, UserWarning, stacklevel=3)

    # Collisions include ancestors, logical aliases and shared tensor storage.
    targets, footprints = [], []
    for entry in entries:
        try:
            module = resolve_module(target, entry["to"])
        except KeyError as error:
            reject(entry, entry["rules"]["on_missing"], str(error))
            targets.append(None)
            continue
        except (ValueError, TypeError) as error:
            reject(entry, "error", f"Invalid target module registry: {error}")
        footprint = _footprint(module)
        for prior, previous in footprints:
            if footprint[0] & previous[0] or footprint[1] & previous[1]:
                reject(entry, "error", f"Target collision: {prior!r} overlaps {entry['to']!r}")
        footprints.append((entry["to"], footprint))
        targets.append(module)

    cache, checkpoint_cache, pending = {}, {}, []
    for entry, module in zip(entries, targets):
        if module is None:
            continue
        spec = config.sources[entry["source"]]
        identity = tuple(str(Path(spec[k]).expanduser().resolve()) if k in spec else None for k in ("checkpoint", "config_path"))
        if identity not in cache:
            try:
                cache[identity] = _load_source(spec, factory, checkpoint_cache)
            except Exception as error:
                cache[identity] = error
        loaded = cache[identity]
        if isinstance(loaded, Exception):
            reject(entry, "error", f"Invalid source checkpoint {entry['checkpoint']}: {loaded}")
        source, provenance = loaded
        report["sources"][entry["source"]] = provenance
        try:
            source_module = resolve_module(source, entry["from"])
        except KeyError as error:
            reject(entry, entry["rules"]["on_missing"], str(error))
            continue
        except (ValueError, TypeError) as error:
            reject(entry, "error", f"Invalid source module registry: {error}")
        try:
            check_module_compatibility(source_module, module)
            if not _tensors(source_module):
                raise ValueError("Module contains no parameters or buffers to transfer")
        except ValueError as error:
            reject(entry, entry["rules"]["on_incompatible"], str(error))
            continue
        pending.append((entry, module, source_module))
    if not pending:
        raise TransferError("Transfer enabled but no mapping succeeded; refusing accidental training from scratch", report)
    # Snapshot all destination tensors: even an unexpected load failure rolls back.
    originals = [(module, copy.deepcopy(module.state_dict()), {k: v.detach().clone() for k, v in _tensors(module).items()})
                 for _, module, _ in pending]
    try:
        for entry, module, source_module in pending:
            module.load_state_dict(copy.deepcopy(source_module.state_dict()), strict=True)
            with torch.no_grad():
                # Includes nonpersistent buffers which state_dict omits.
                for name, value in _tensors(module).items():
                    value.copy_(_tensors(source_module)[name])
            entry.update(status="transferred", reason="Strict architecture and tensor checks passed",
                         num_parameters=sum(p.numel() for p in module.parameters()))
            _LOG.info("Transferred %s:%s -> %s (%s parameters)", entry["source"], entry["from"], entry["to"], entry["num_parameters"])
    except Exception as error:
        for module, state, tensors in originals:
            module.load_state_dict(state, strict=True)
            with torch.no_grad():
                for name, value in _tensors(module).items():
                    value.copy_(tensors[name])
        for entry, _, _ in pending:
            entry.update(status="error", reason=f"Transfer rolled back: {error}", num_parameters=0)
        raise TransferError(f"Transfer rolled back: {error}", report) from error
    return report
