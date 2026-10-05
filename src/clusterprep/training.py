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
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from .dataset import ClusterDataset
from .io import CLASS_NAMES
from .metrics import classification_metrics
from .models import build_model, _list_models_with_scattering
from .splits import split_data_keys
from .processed import prepare_processed
from .progress import progress, message
from .training_core import (TrainingConfig, LEGACY_TRAINING_CONFIG, EarlyStopping,
    compute_class_weights, build_criterion, build_optimizer, build_scheduler,
    mixup_batch, move_batch_to_device, forward_model)


@dataclass(frozen=True)
class ExperimentConfig(TrainingConfig):
    name: str
    mode: str = "raw"
    model: str = "cnn"
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
        if not isinstance(self.model_params, dict) or not all(isinstance(k, str) for k in self.model_params):
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
        if self.size < 2 or not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("size >= 2 et num_workers >= 0 requis")
        for key in ("learning_rate", "weight_decay", "min_delta"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} doit être fini et positif ou nul")
        if self.learning_rate == 0:
            raise ValueError("learning_rate doit être strictement positif")
        if self.monitor not in {"val_loss", "val_macro_f1", "val_balanced_accuracy"}:
            raise ValueError("monitor attendu : val_loss, val_macro_f1, val_balanced_accuracy")
        if self.class_weight not in {"none", "balanced"}:
            raise ValueError("class_weight doit être none ou balanced")


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


def _validate_splits(info, splits, class_names):
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
            if info[key]["class"] not in class_names:
                raise ValueError(f"Classe inconnue : {key}")
            clusters.add(info[key]["cluster_id"])
        if seen & clusters:
            raise ValueError("Fuite de clusters entre les splits")
        seen.update(clusters)
    labels = {}
    for keys in splits:
        for key in keys:
            item = info[key]
            previous = labels.setdefault(item["cluster_id"], item["class"])
            if previous != item["class"]:
                raise ValueError("Classes contradictoires pour un cluster")


def _inputs(batch, mode, model=None):
    names = (("raw",) if mode == "raw" else ("raw", "chandra", "chandra_mask"))
    x = torch.cat([batch[name] for name in names],dim=1)
    if (model is not None and getattr(model,"uses_precomputed_scattering",False,)):
        if "scattering" not in batch:
            raise ValueError("Le modèle requiert un scattering précalculé")
        return x, batch["scattering"]
    return x


def _epoch(model, loader, config, device, class_names, optimizer=None, weights=None, description=None,
           criterion=None, scheduler=None, input_adapter=None):
    training = optimizer is not None
    model.train(training)
    labels, probabilities, predictions = [], [], []
    total_loss, count = 0.0, 0
    criterion = criterion or nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)
    with (torch.enable_grad() if training else torch.inference_mode()), progress(
            loader, description or ("Entraînement" if training else "Évaluation"), getattr(config, "verbose", False)) as bar:
        for batch in bar:
            batch = move_batch_to_device(batch, device)
            target = batch["label"]
            if training:
                optimizer.zero_grad(set_to_none=True)
            inputs = input_adapter(batch) if input_adapter else _inputs(batch, getattr(config, "mode", "raw"), model)
            mixed = training and config.use_mixup and random.random() < config.mixup_probability
            if mixed:
                inputs, target_a, target_b, lam = mixup_batch(inputs, target, config.mixup_alpha)
            logits = forward_model(model, inputs)
            if logits.shape != (len(target), len(class_names)) or not torch.isfinite(logits).all():
                raise ValueError("Le modèle doit retourner des logits finis [B, nombre_classes]")
            # CE simple conservée séparément pour les comparaisons de cohortes.
            losses = nn.functional.cross_entropy(logits, target, reduction="none")
            loss = (lam * criterion(logits, target_a) + (1 - lam) * criterion(logits, target_b)
                    if mixed else criterion(logits, target))
            if not torch.isfinite(loss):
                raise ValueError("Perte non finie")
            if training:
                loss.backward()
                if config.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            proba = logits.detach().softmax(dim=1).cpu().numpy()
            truth = target.cpu().numpy()
            total_loss += loss.item() * len(target)
            count += len(target)
            bar.set_postfix(loss=f"{total_loss / count:.4f}", refresh=False)
            labels.extend(truth.tolist())
            probabilities.extend(proba.tolist())
            if not training:
                sample_losses = losses.detach().cpu().tolist()
                for i in range(len(target)):
                    predictions.append({"key": batch.get("key", list(range(count - len(target), count)))[i],
                        "cluster_id": batch.get("cluster_id", [None] * len(target))[i], "label": int(truth[i]),
                        "prediction": int(proba[i].argmax()),
                        "has_chandra": bool(batch.get("has_chandra", [False] * len(target))[i]),
                        "loss": sample_losses[i],
                        "probabilities": proba[i].tolist()})
    if count == 0:
        raise ValueError("Aucun exemple dans le DataLoader")
    metrics = classification_metrics(labels, probabilities, class_names)
    metrics["loss"] = total_loss / count
    return metrics, predictions


class Trainer:
    """Moteur générique ; input_adapter(batch) sélectionne les entrées du modèle.

    L'adaptateur renvoie un tenseur, un tuple d'arguments ou un dict d'arguments
    nommés. Toutes ces entrées partagent le même MixUp. Les batches contiennent
    au minimum « label » ; les métadonnées de prédiction sont facultatives.
    """
    def __init__(self, model, config, class_names, class_counts, *, device="cpu",
                 steps_per_epoch, input_adapter=None):
        self.model = model.to(device)
        self.config, self.class_names, self.device = config, tuple(class_names), device
        self.input_adapter = input_adapter
        self.criterion = build_criterion(config, class_counts, device)
        self.optimizer = build_optimizer(model, config)
        self.scheduler = build_scheduler(self.optimizer, config, steps_per_epoch)
        monitor = getattr(config, "monitor", "val_loss")
        self.stopping = EarlyStopping(config.patience, getattr(config, "min_delta", 0.0),
                                      "min" if monitor == "val_loss" else "max")
        self.best_epoch = 0
        self.stopped_early = False

    def train_one_epoch(self, loader, description=None):
        return _epoch(self.model, loader, self.config, self.device, self.class_names,
                      optimizer=self.optimizer, criterion=self.criterion,
                      scheduler=self.scheduler, input_adapter=self.input_adapter,
                      description=description)

    def evaluate(self, loader, description=None):
        return _epoch(self.model, loader, self.config, self.device, self.class_names,
                      criterion=self.criterion, input_adapter=self.input_adapter,
                      description=description)

    def fit(self, train_loader, val_loader):
        """Entraîne puis restaure le meilleur modèle, même sans arrêt anticipé.

        Créer un Trainer et un modèle neufs pour chaque expérience.
        """
        history = []
        for epoch in range(1, self.config.epochs + 1):
            train, _ = self.train_one_epoch(train_loader)
            validation, _ = self.evaluate(val_loader)
            history.append({"epoch": epoch, "train": train, "validation": validation})
            key = getattr(self.config, "monitor", "val_loss").removeprefix("val_")
            improved, stop = self.stopping.update(validation[key], self.model)
            if improved:
                self.best_epoch = epoch
            if stop and self.config.use_early_stopping:
                self.stopped_early = True
                break
        self.stopping.restore(self.model)
        return history

    def predict(self, loader):
        """Probabilités CPU, dans l'ordre du loader ; labels non nécessaires."""
        self.model.eval()
        predictions = []
        with torch.inference_mode():
            for batch in loader:
                batch = move_batch_to_device(batch, self.device)
                inputs = (self.input_adapter(batch) if self.input_adapter else
                          _inputs(batch, getattr(self.config, "mode", "raw"), self.model))
                predictions.append(forward_model(self.model, inputs).softmax(1).cpu())
        if not predictions:
            raise ValueError("Aucun exemple dans le DataLoader")
        return torch.cat(predictions)


def _cohort(predictions, class_names, sources):
    if not predictions:
        return {"cohort_id": None, "metrics": None}
    ordered = sorted(predictions, key=lambda p: p["key"])
    metrics = classification_metrics([p["label"] for p in ordered],
                                     [p["probabilities"] for p in ordered], class_names)
    metrics["loss"] = float(np.mean([p["loss"] for p in ordered]))
    identity = {"classes": list(class_names),
                "samples": [(p["key"], p["cluster_id"], p["label"], sources[p["key"]])
                            for p in ordered]}
    return {"cohort_id": _fingerprint(identity), "metrics": metrics}


def _source_signature(item):
    signature = {}
    for name in ("raw_path", "chandra_path"):
        if item.get(name):
            path = Path(item[name]).resolve()
            stat = path.stat()
            signature[name] = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        else:
            signature[name] = None
    return signature


def _write_predictions(path, predictions, class_names):
    base_fields = ["key", "cluster_id", "label", "prediction", "has_chandra", "loss"]
    probability_fields = [f"probability_{name}" for name in class_names]
    fields = base_fields + probability_fields
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for prediction in predictions:
            row = {key: prediction[key] for key in base_fields}
            row.update(zip(probability_fields, prediction["probabilities"]))
            writer.writerow(row)


def run_experiment(data_info, config, output_dir="results/experiments", *,
                   class_names=CLASS_NAMES, splits=None, model_factory=None, input_adapter=None):
    """Entraîne un modèle neuf, restaure son meilleur état, puis évalue le test une fois.

    model_factory(config, num_classes) doit renvoyer un nn.Module prenant
    un tenseur [B, 1 ou 3, H, W] et retournant des logits [B, num_classes].
    input_adapter(batch) peut remplacer cette entrée par un tenseur, tuple ou dict.
    Chaque appel crée un répertoire unique et ne remplace aucune expérience.
    """
    class_names = tuple(class_names)
    if splits is None:
        splits = split_data_keys(data_info, mode="raw", seed=42)
    splits = tuple(list(keys) for keys in splits)
    _validate_splits(data_info, splits, class_names)
    base_splits = splits
    if config.mode == "pair":
        splits = tuple([k for k in keys if data_info[k].get("chandra_path")] for keys in splits)
    _validate_splits(data_info, splits, class_names)
    class_counts = [sum(data_info[k]["class"] == name for k in splits[0]) for name in class_names]
    if min(class_counts) == 0:
        raise ValueError("Chaque classe doit être présente dans l'entraînement après filtrage")
    run_id = f"{config.name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _json(run_dir / "status.json", {"status": "running"})
    try:
        if config.model in _list_models_with_scattering() and config.use_mixup:
            print("⚠️  Modèle avec scattering : MixUp désactivé pour cette expérience.")
            config = replace(config, use_mixup=False)
        return _run(data_info, config, run_dir, class_names, splits, base_splits,
                    class_counts, model_factory or build_model, input_adapter)
    except Exception as error:
        _json(run_dir / "status.json", {"status": "failed", "error": str(error)})
        raise


def _run(info, config, run_dir, class_names, splits, base_splits, class_counts, factory, input_adapter=None):
    started = time.perf_counter()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = factory(config, len(class_names)).to(device)
    needs_multiple = any(isinstance(layer, nn.modules.batchnorm._BatchNorm)
                         for layer in model.modules())
    train_count = len(splits[0]) * (ClusterDataset.N_VARIANTS if config.augment else 1)
    if needs_multiple and (config.batch_size < 2 or train_count < 2):
        raise ValueError("Un modèle avec BatchNorm requiert batch_size >= 2 et au moins deux exemples train")
    message(f"\n{config.name} | mode={config.mode}, modèle={config.model}, device={device}, "
            f"seed={config.seed}\n"
            f"Train/validation/test : {'/'.join(str(len(k)) for k in splits)} amas ; "
            f"augmentation={'24 variantes' if config.augment else 'non'} ; "
            f"{config.epochs} époques maximum, early stopping={config.monitor} "
            f"(patience={config.patience}).\nRésultats : {run_dir}", config.verbose)
    preparation = None
    if config.processed_dir is not None:
        preparation = prepare_processed(info, [k for keys in splits for k in keys],
            config.mode, config.size, config.processed_dir, config.verbose,
            warning_log=run_dir / "fits_warnings.log")
        _json(run_dir / "preprocessing.json", preparation)
    scattering_dir = None
    scattering_params = None
    if getattr(model,"uses_precomputed_scattering",False):
        if config.use_mixup:
            raise ValueError(
                "Le scattering précalculé n'est pas compatible "
                "exactement avec le MixUp d'entrée. "
                "Utiliser use_mixup=False pour DualSSN."
            )
        from .scattering import prepare_scattering
        scattering_params = model.scattering_config
        scattering_dir = (Path(config.processed_dir or "processed") / "scattering")
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
            **scattering_params,
        )
        _json(
            run_dir / "scattering.json",
            scattering_preparation,
        )
    datasets = [ClusterDataset(info, keys, class_names, mode=config.mode, size=config.size,
                 augment=config.augment and i == 0, cache=config.cache,
                 processed_dir=config.processed_dir, scattering_dir=scattering_dir,
                 scattering_params=scattering_params)
                for i, keys in enumerate(splits)]
    message(f"Exemples par split : {[len(ds) for ds in datasets]}", config.verbose)
    loaders = []
    for i, ds in enumerate(datasets):
        generator = torch.Generator().manual_seed(config.seed + i)
        batching = {"batch_size": config.batch_size, "shuffle": i == 0,
                    "drop_last": i == 0 and config.drop_last_train}
        if i == 0 and needs_multiple and not config.drop_last_train:
            batching = {"batch_sampler": _MergeSingletonBatchSampler(
                RandomSampler(ds, generator=generator), config.batch_size, drop_last=False)}
        loaders.append(DataLoader(ds, **batching, num_workers=config.num_workers,
            worker_init_fn=_seed_worker, generator=generator,
            persistent_workers=config.num_workers > 0))
    trainer = Trainer(model, config, class_names, class_counts, device=device,
                      steps_per_epoch=len(loaders[0]), input_adapter=input_adapter)
    optimizer = trainer.optimizer
    metadata = {"schema_version": 2, "config": asdict(config), "class_names": list(class_names),
        "train_class_counts": dict(zip(class_names, class_counts)),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_repr": repr(model),
        "library_source_sha256": _fingerprint({p.name: p.read_text() for p in
                                               sorted(Path(__file__).parent.glob("*.py"))}),
        "parameters": sum(p.numel() for p in model.parameters()),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "numpy", "astropy", "reproject", "scikit-learn")},
        "normalization": {"percentile_lo": 30, "percentile_hi": 99, "alpha": 10.0},
        "created_at": datetime.now(timezone.utc).isoformat()}
    _json(run_dir / "config.json", metadata)
    split_manifest = {name: keys for name, keys in zip(("train", "validation", "test"), splits)}
    split_manifest["base_splits"] = dict(zip(("train", "validation", "test"), base_splits))
    split_manifest["samples"] = {k: {**info[k], "raw_path": str(Path(info[k]["raw_path"]).resolve()),
        "chandra_path": str(Path(info[k]["chandra_path"]).resolve()) if info[k].get("chandra_path") else None}
        for keys in base_splits for k in keys}
    sources = {k: _source_signature(info[k]) for keys in base_splits for k in keys}
    split_manifest["source_signatures"] = sources
    _json(run_dir / "splits.json", split_manifest)
    stopping = trainer.stopping
    history, best_epoch, stopped = [], 0, False
    history_path = run_dir / "history.csv"
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics, _ = trainer.train_one_epoch(loaders[0],
                                  description=f"Époque {epoch}/{config.epochs} · train")
        val_metrics, _ = trainer.evaluate(loaders[1],
                                description=f"Époque {epoch}/{config.epochs} · validation")
        record = {"epoch": epoch, "seconds": time.perf_counter() - epoch_start,
                  "learning_rate": optimizer.param_groups[0]["lr"],
                  "train": train_metrics, "validation": val_metrics}
        history.append(record)
        with (run_dir / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        flat = {"epoch": epoch, "seconds": record["seconds"], "learning_rate": record["learning_rate"]}
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            flat.update({f"{prefix}_{key}": value for key, value in metrics.items()
                         if not isinstance(value, (dict, list))})
        with history_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(flat)
        improved, stopped = stopping.step(val_metrics[config.monitor.removeprefix("val_")], model)
        stopped = stopped and config.use_early_stopping
        if improved:
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler else None,
                "config": asdict(config), "class_names": list(class_names), "epoch": epoch,
                "monitor": config.monitor, "score": stopping.best}, run_dir / "best.pt")
        message(f"Époque {epoch:03d}/{config.epochs} | loss train={train_metrics['loss']:.4f} "
                f"val={val_metrics['loss']:.4f} | F1 val={val_metrics['macro_f1']:.3f} | "
                f"{'checkpoint enregistré' if improved else f'patience {stopping.bad_epochs}/{config.patience}'} "
                f"| {record['seconds']:.1f} s", config.verbose)
        if stopped:
            message("Arrêt anticipé.", config.verbose)
            break
    message(f"Restauration du checkpoint époque {best_epoch} ; évaluation finale.", config.verbose)
    stopping.restore(model)
    val_metrics, val_predictions = trainer.evaluate(loaders[1],
                                           description="Validation finale")
    test_metrics, test_predictions = trainer.evaluate(loaders[2],
                                             description="Test final")
    _write_predictions(run_dir / "validation_predictions.csv", val_predictions, class_names)
    _write_predictions(run_dir / "test_predictions.csv", test_predictions, class_names)
    results = {"schema_version": 2, "run_id": run_dir.name, "config": asdict(config),
        "class_names": list(class_names), "best_epoch": best_epoch, "epochs_ran": len(history),
        "stopped_early": stopped, "monitor_best": stopping.best,
        "seconds": time.perf_counter() - started,
        "preprocessing": preparation,
        "split_counts": dict(zip(("train", "validation", "test"), map(len, splits))),
        "validation": val_metrics, "test": test_metrics,
        "test_cohorts": {"all": _cohort(test_predictions, class_names, sources),
            "paired": _cohort([p for p in test_predictions if p["has_chandra"]], class_names, sources),
            "raw_only": _cohort([p for p in test_predictions if not p["has_chandra"]], class_names, sources)}}
    _json(run_dir / "metrics.json", results)
    _json(run_dir / "status.json", {"status": "completed"})
    message(f"Terminé : test loss={test_metrics['loss']:.4f}, "
            f"F1 macro={test_metrics['macro_f1']:.3f}, accuracy={test_metrics['accuracy']:.3f} "
            f"| {results['seconds']:.1f} s\nFichiers : {run_dir}", config.verbose)
    return run_dir


def run_experiments(data_info, configs, output_dir="results/experiments", *,
                    class_names=CLASS_NAMES, splits=None, split_seed=42,
                    test_size=.2, val_size=.2, n_folds=None, fold_index=0, model_factory=None, input_adapter=None):
    """Suite séquentielle partageant les splits, indépendamment des graines modèle."""
    configs = list(configs)
    if not configs:
        raise ValueError("Au moins une configuration d'expérience requise")
    if splits is None:
        splits = split_data_keys(data_info, mode="raw", seed=split_seed, test_size=test_size,
                                val_size=val_size, n_folds=n_folds, fold_index=fold_index)
    runs = []
    for index, config in enumerate(configs, 1):
        message(f"\nExpérience {index}/{len(configs)}", config.verbose)
        runs.append(run_experiment(data_info, config, output_dir, class_names=class_names,
                                   splits=splits, model_factory=model_factory, input_adapter=input_adapter))
    return runs


def compare_experiments(run_dirs, output_path=None, cohort="paired"):
    """Table de comparaison ; refuse les cohortes ou classes de test différentes."""
    if cohort not in {"all", "paired", "raw_only"}:
        raise ValueError("Cohorte attendue : all, paired ou raw_only")
    rows, cohort_id = [], None
    for directory in run_dirs:
        result = json.loads((Path(directory) / "metrics.json").read_text())
        selected = result["test_cohorts"][cohort]
        if selected["cohort_id"] is None:
            raise ValueError(f"Cohorte vide pour {directory}")
        if cohort_id is not None and cohort_id != selected["cohort_id"]:
            raise ValueError("Cohortes de test différentes : comparaison directe refusée")
        cohort_id = selected["cohort_id"]
        row = {"run_id": result["run_id"], "name": result["config"]["name"],
               "mode": result["config"]["mode"], "model": result["config"]["model"],
               "seed": result["config"]["seed"], "best_epoch": result["best_epoch"],
               "seconds": result["seconds"], "cohort": cohort, "cohort_id": cohort_id}
        row.update({k: v for k, v in selected["metrics"].items() if not isinstance(v, (dict, list))})
        rows.append(row)
    if not rows:
        raise ValueError("Au moins une expérience requise")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows
