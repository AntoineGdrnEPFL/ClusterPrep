"""Internal cluster-level CV; selection never receives the external test loader.

The optimization engine remains Trainer/run_experiment. This module only owns
partitions, search, aggregation, initialization provenance and final refitting.
"""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import platform
import json
from pathlib import Path
import uuid
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .labels import task_target
from .splits import eligible_for_task
from .transfer import TransferConfig
from .modular_transfer import ModularTransferConfig


@dataclass(frozen=True)
class CrossValidationConfig:
    enabled: bool = False
    n_splits: int = 5
    shuffle: bool = True
    seed: int = 42
    strategy: str = "auto"
    selection_metric: str = "val_loss"
    selection_mode: str = "min"
    max_epochs: int | None = None  # None uses the existing experiment budget.
    phase_durations: list | None = None  # Complete {phase_name: epochs} candidates.
    retrain_full_train: bool = True
    evaluate_test_after_retrain: bool = True
    source_provenance_policy: str = "error"

    def __post_init__(self):
        for key in ("enabled", "shuffle", "retrain_full_train", "evaluate_test_after_retrain"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"cross_validation.{key} must be boolean")
        if not self.enabled:
            return
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, int) or self.n_splits < 2:
            raise ValueError("cross_validation.n_splits must be >= 2")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("cross_validation.seed must be an integer in [0, 2**32)")
        if self.strategy not in {"auto", "stratified", "multilabel"}:
            raise ValueError("CV strategy: auto, stratified or multilabel")
        if self.selection_metric not in {"val_loss", "val_macro_f1", "f1_macro"}:
            raise ValueError("CV selection_metric: val_loss, val_macro_f1 or f1_macro")
        if self.selection_mode not in {"min", "max"}:
            raise ValueError("CV selection_mode: min or max")
        if self.max_epochs is not None and (
            isinstance(self.max_epochs, bool) or not isinstance(self.max_epochs, int) or self.max_epochs < 1
        ):
            raise ValueError("CV max_epochs must be >= 1")
        if self.source_provenance_policy not in {"error", "warn"}:
            raise ValueError("CV source_provenance_policy: error or warn")
        if self.phase_durations is not None:
            if not isinstance(self.phase_durations, list) or not self.phase_durations:
                raise ValueError("CV phase_durations must be a nonempty list of objects")
            for candidate in self.phase_durations:
                if not isinstance(candidate, dict) or not candidate or any(
                    not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, int) or v < 1
                    for k, v in candidate.items()
                ):
                    raise ValueError("Each CV phase duration must be a positive integer keyed by phase name")


def parse_cross_validation(value):
    if isinstance(value, dict):
        try:
            value = CrossValidationConfig(**value)
        except TypeError as error:
            raise ValueError(f"Invalid cross_validation: {error}") from error
    if value is not None and not isinstance(value, CrossValidationConfig):
        raise ValueError("cross_validation must be an object or null")
    return value


def _read(path):
    return json.loads(Path(path).read_text())


def _dump(path, value):
    # Same atomic JSON writer as all ordinary experiments.
    from .training import _json
    _json(Path(path), value)


def label_distribution(info, keys, config):
    targets = [task_target(info[k], config.task, config.candidate_policy)[0] for k in keys]
    n = len(keys)
    if config.task == "de_nde":
        counts = {"DE": targets.count(0), "NDE": targets.count(1)}
        return {"n_clusters": n, **counts, "proportions": {k: v / n if n else None for k, v in counts.items()}}
    y = np.asarray(targets).reshape(-1, 2)
    known = np.asarray([task_target(info[k], config.task, config.candidate_policy)[1] for k in keys]).reshape(-1, 2).astype(bool)
    positive = (y == 1) & known
    negative = (y == 0) & known
    counts = {"RR+": int(positive[:, 0].sum()), "RH+": int(positive[:, 1].sum()),
              "RR+RH+": int(positive.all(1).sum()), "RR-RH-": int(negative.all(1).sum())}
    return {"n_clusters": n, **counts,
            "known_negative": dict(zip(("RR", "RH"), map(int, negative.sum(0)))),
            "unknown": dict(zip(("RR", "RH"), map(int, (~known).sum(0)))),
            "proportions": {k: v / n if n else None for k, v in counts.items()}}


def build_internal_folds(info, keys, config):
    """Split unique eligible clusters, preserving unknown labels as unknown.

    RR/RH uses the required two-column indicator matrix on fully known rows.
    Partially known rows are allocated by seed/size without interpreting their
    unknown entries as negatives; they retain their masks during training.
    """
    cv = config.cross_validation
    strategy = "stratified" if config.task == "de_nde" else "multilabel"
    if cv.strategy not in {"auto", strategy}:
        raise ValueError(f"CV strategy {cv.strategy} is incompatible with task {config.task}")
    keys = sorted(keys, key=lambda k: str(info[k]["cluster_id"]))
    ids = [str(info[k]["cluster_id"]) for k in keys]
    if len(ids) != len(set(ids)):
        raise ValueError("CV requires unique cluster IDs before augmentation")
    if len(keys) < cv.n_splits:
        raise ValueError("Too few clusters: reduce n_splits (e.g. 5 to 3)")
    if any(not eligible_for_task(info[k], config.task, config.candidate_policy) for k in keys):
        raise ValueError("Non-eligible cluster in the external training pool")
    diagnostics = []
    distribution = label_distribution(info, keys, config)
    rng_state = cv.seed if cv.shuffle else None
    if config.task == "de_nde":
        y = np.asarray([task_target(info[k], config.task, config.candidate_policy)[0] for k in keys])
        if min(distribution['DE'], distribution['NDE']) < cv.n_splits:
            raise ValueError("Too few clusters per class for CV: reduce n_splits (e.g. 5 to 3)")
        splitter = StratifiedKFold(cv.n_splits, shuffle=cv.shuffle, random_state=rng_state)
        validation_indices = [list(b) for _, b in splitter.split(np.zeros(len(keys)), y)]
        partial = []
    else:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        targets = [task_target(info[k], config.task, config.candidate_policy) for k in keys]
        y = np.asarray([t[0] for t in targets])
        known = np.asarray([t[1] for t in targets]).astype(bool)
        complete = np.flatnonzero(known.all(1))
        partial = np.flatnonzero(~known.all(1)).tolist()
        if len(complete) < cv.n_splits:
            raise ValueError("Too few fully known RR/RH clusters: reduce n_splits; unknown labels cannot be treated as negatives")
        for label in ("RR", "RH"):
            if distribution[label + '+'] < cv.n_splits:
                diagnostics.append(f"{label}: positives < n_splits; reduce n_splits (e.g. 5 to 3)")
        splitter = MultilabelStratifiedKFold(cv.n_splits, shuffle=cv.shuffle, random_state=rng_state)
        # iterstrat resolves ties randomly even with shuffle=False. Seed that
        # operation locally without advancing the caller's NumPy RNG.
        previous_rng = np.random.get_state()
        try:
            np.random.seed(cv.seed)
            validation_indices = [complete[b].tolist() for _, b in splitter.split(np.zeros(len(complete)), y[complete])]
        finally:
            np.random.set_state(previous_rng)
        if partial:
            diagnostics.append("Partially known labels allocated by size/seed, not encoded as negatives for stratification")
            order = np.random.default_rng(cv.seed).permutation(partial) if cv.shuffle else partial
            for index in order:
                destination = min(range(cv.n_splits), key=lambda i: (len(validation_indices[i]), i))
                validation_indices[destination].append(int(index))
    folds = []
    for i, val_indices in enumerate(validation_indices):
        selected = set(val_indices)
        train = [k for j, k in enumerate(keys) if j not in selected]
        validation = [k for j, k in enumerate(keys) if j in selected]
        train_dist = label_distribution(info, train, config)
        val_dist = label_distribution(info, validation, config)
        if not train or not validation:
            raise ValueError("Empty CV fold: reduce n_splits")
        if config.task == "de_nde":
            if not train_dist['DE'] or not train_dist['NDE']:
                raise ValueError("CV sub-train lacks a class: reduce n_splits")
        else:
            for label in ("RR", "RH"):
                if not train_dist[label + '+'] or not train_dist['known_negative'][label]:
                    raise ValueError(f"Fold {i}: no known positive/negative {label} in sub-train; reduce n_splits")
                if not val_dist[label + '+'] or not val_dist['known_negative'][label]:
                    diagnostics.append(f"Fold {i}: {label} ROC-AUC undefined (missing known class); reduce n_splits")
        folds.append({"fold": i, "train": train, "validation": validation,
                      "train_cluster_ids": [str(info[k]['cluster_id']) for k in train],
                      "validation_cluster_ids": [str(info[k]['cluster_id']) for k in validation],
                      "distribution": {"train": train_dist, "validation": val_dist}})
    for diagnostic in diagnostics:
        warnings.warn(diagnostic, UserWarning)
    return {"strategy": type(splitter).__name__, "seed": cv.seed, "shuffle": cv.shuffle,
            "distribution": distribution, "partially_known_clusters": [ids[i] for i in partial],
            "warnings": diagnostics, "folds": folds}


def aggregate_curves(histories, sizes, metric="val_loss", mode="min"):
    """Weighted means and population std; epoch numbers are one-based."""
    if metric not in {"val_loss", "val_macro_f1", "f1_macro"} or mode not in {"min", "max"}:
        raise ValueError("Invalid CV aggregation metric/mode")
    key = "loss" if metric == "val_loss" else "macro_f1"
    if not histories or len(histories) != len(sizes) or any(n <= 0 for n in sizes):
        raise ValueError("Nonempty CV histories and positive fold sizes required")
    length = len(histories[0])
    if not length or any(len(h) != length for h in histories):
        raise ValueError("Incomplete CV curves: all folds must run the common epoch budget")
    curves = []
    weights = np.asarray(sizes, dtype=float)
    for epoch in range(1, length + 1):
        records = [h[epoch - 1] for h in histories]
        if any(r['epoch'] != epoch for r in records):
            raise ValueError("CV epochs must be contiguous and one-based")
        values = [(r.get('validation') or {}).get(key) for r in records]
        if any(v is None or not np.isfinite(v) for v in values):
            raise ValueError(f"Undefined {metric} at CV epoch {epoch}; reduce n_splits or use val_loss")
        mean = float(np.average(values, weights=weights))
        std = float(np.sqrt(np.average((np.asarray(values) - mean)**2, weights=weights)))
        curves.append({"epoch": epoch, "mean": mean, "std": std,
                       "fold_values": values, "weights": list(map(int, sizes))})
    chosen = min(curves, key=lambda r: (r['mean'] if mode == 'min' else -r['mean'], r['epoch']))
    return {"metric": metric, "mode": mode, "epoch_indexing": "one-based",
            "weighting": "validation cluster counts", "dispersion": "weighted population std",
            "tie_break": "earliest epoch", "selected_epoch": chosen['epoch'],
            "score": chosen['mean'], "curves": curves}


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _sources(transfer):
    if isinstance(transfer, TransferConfig):
        return [{"checkpoint": str(Path(transfer.source_experiment) / 'best.pt'),
                 "config_path": str(Path(transfer.source_experiment) / 'config.json')}]
    if isinstance(transfer, ModularTransferConfig) and transfer.enabled:
        return list(transfer.sources.values())
    return []


def audit_initialization(config, info, train, test, folds):
    """Inspect source manifests recursively, including earlier transfers.

    Missing provenance is never considered evidence of external pretraining.
    Any known source exposure to the final test is always fatal.
    """
    from .modular_transfer import parse_transfer
    import torch
    seen, entries, problems = set(), [], []
    train_ids = {str(info[k]['cluster_id']) for k in train}
    test_ids = {str(info[k]['cluster_id']) for k in test}
    pending = _sources(config.transfer)
    while pending:
        source = pending.pop(0)
        checkpoint = Path(source['checkpoint']).expanduser().resolve()
        identity = (str(checkpoint), source.get('config_path'))
        if identity in seen:
            continue
        seen.add(identity)
        entry = {"checkpoint": str(checkpoint), "sha256": _digest(checkpoint)}
        entries.append(entry)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if source.get('config_path'):
            metadata_path = Path(source['config_path']).expanduser().resolve()
            entry['config_path'] = str(metadata_path)
            entry['config_sha256'] = _digest(metadata_path)
            metadata = _read(metadata_path)
        else:
            metadata = payload
        raw = metadata.get('config', metadata) if isinstance(metadata, dict) else {}
        # Match source reconstruction: a bare state_dict is not a configuration.
        if (not isinstance(raw, dict) or not {'model', 'mode', 'size'} <= raw.keys()) and (checkpoint.parent / 'config.json').is_file():
            metadata_path = checkpoint.parent / 'config.json'
            metadata = _read(metadata_path)
            entry.update(config_path=str(metadata_path), config_sha256=_digest(metadata_path))
            raw = metadata.get('config', metadata)
        if not isinstance(raw, dict) or not {'model', 'mode', 'size'} <= raw.keys():
            problems.append(f"Source {checkpoint}: missing configuration; ancestral transfer exposure cannot be verified")
            raw = {}
        manifest_path = checkpoint.parent / 'splits.json'
        embedded = payload.get('split_provenance') if isinstance(payload, dict) else None
        if embedded is None and isinstance(metadata, dict):
            embedded = metadata.get('split_provenance')
        if not manifest_path.is_file() and not isinstance(embedded, dict):
            entry['provenance'] = 'unknown'
            problems.append(f"Source {checkpoint}: missing splits.json; pretraining exposure cannot be verified")
        else:
            manifest = _read(manifest_path) if manifest_path.is_file() else embedded
            if manifest_path.is_file():
                entry.update(manifest=str(manifest_path), manifest_sha256=_digest(manifest_path))
            exposure = set()
            for split in ('train', 'validation'):
                if split not in manifest or not isinstance(manifest[split], list):
                    raise ValueError(f"Invalid source provenance: {manifest_path}/{split}")
                for key in manifest[split]:
                    cid = manifest.get('samples', {}).get(key, {}).get('cluster_id')
                    if cid is None:
                        raise ValueError(f"Missing source cluster_id: {manifest_path}/{key}")
                    exposure.add(str(cid))
            entry.update(provenance='manifest' if manifest_path.is_file() else 'embedded',
                         exposed_cluster_ids=sorted(exposure),
                         external_test_overlap=sorted(exposure & test_ids),
                         external_train_overlap=sorted(exposure & train_ids),
                         fold_validation_overlap=[sorted(exposure & set(f['validation_cluster_ids'])) for f in folds])
            if exposure & test_ids:
                problems.append(f"Pretraining leakage into external test from {checkpoint}: {sorted(exposure & test_ids)}")
            if exposure & train_ids:
                problems.append(f"Source {checkpoint}: pretraining used internal validation clusters; use external pretraining or retrain it inside each fold")
        pending.extend(_sources(parse_transfer(raw.get('transfer'))))
        # Reconstruction overrides must not hide the weights' recorded ancestry.
        if isinstance(payload, dict) and isinstance(payload.get('config'), dict):
            pending.extend(_sources(parse_transfer(payload['config'].get('transfer'))))
        # A final model from earlier CV has also used all its external train for selection.
        context = (payload.get('cross_validation') or {}) if isinstance(payload, dict) else {}
        for prior in (context.get('initialization_audit') or {}).get('sources', []):
            pending.append({k: prior[k] for k in ('checkpoint', 'config_path') if k in prior})
    report = {"sources": entries, "warnings": problems, "verified_no_overlap": not problems,
              "policy": config.cross_validation.source_provenance_policy,
              "external_test_leakage": any(e.get("external_test_overlap") for e in entries)}
    return report


def _check_sources(report):
    for source in report['sources']:
        for path_key, digest_key in (('checkpoint', 'sha256'), ('config_path', 'config_sha256'), ('manifest', 'manifest_sha256')):
            if path_key in source and _digest(source[path_key]) != source[digest_key]:
                raise ValueError(f"CV source changed during search: {source[path_key]}")


def _with_durations(config, durations):
    from .training import _training_phases
    phases = [replace(p, epochs=durations[p.name]) for p in _training_phases(config)]
    if isinstance(config.transfer, TransferConfig):
        return replace(config, transfer=replace(config.transfer, phases=phases))
    return replace(config, training_phases=phases)


def _plot_curves(path, trials, phased):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(8, 4), layout='constrained')
    ax = fig.subplots()
    for i, trial in enumerate(trials):
        curves = trial['aggregation']['curves']
        epochs = [r['epoch'] for r in curves]
        means = np.array([r['mean'] for r in curves])
        std = np.array([r['std'] for r in curves])
        ax.plot(epochs, means, label=f"Trial {i}: {trial['durations']}" if phased else 'CV mean')
        ax.fill_between(epochs, means - std, means + std, alpha=.15)
    ax.set(xlabel='Epoch (1-based)', ylabel=trials[0]['aggregation']['metric'])
    ax.legend()
    fig.savefig(path, dpi=130)


def run_cross_validation(info, config, output_dir, *, splits, class_names,
                         model_factory=None, input_adapter=None, split_metadata=None):
    from .training import run_experiment, _training_phases, _validate_splits, _source_signature
    cv = config.cross_validation
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    root = Path(output_dir) / f'{config.name}-cv-{stamp}-{uuid.uuid4().hex[:8]}'
    root.mkdir(parents=True, exist_ok=False)
    (root / 'cv').mkdir()
    _dump(root / 'status.json', {'status': 'running', 'stage': 'preflight'})
    _dump(root / 'config_resolved.json', asdict(config))
    _dump(root / 'environment.json', {
        'python': platform.python_version(),
        'versions': {name: importlib.metadata.version(name) for name in (
            'torch', 'torchvision', 'numpy', 'astropy', 'reproject', 'scikit-learn', 'iterative-stratification')},
        'source_sha256': {p.name: _digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
    })
    try:
        # Both historical development partitions belong to the external training pool.
        # Eligibility filters never move a cluster across the external boundary.
        original = {name: list(keys) for name, keys in zip(('train', 'validation', 'test'), splits)}
        eligible = lambda k: eligible_for_task(info[k], config.task, config.candidate_policy) and (
            config.mode != 'pair' or info[k].get('chandra_path'))
        train = [k for k in splits[0] + splits[1] if eligible(k)]
        test = [k for k in splits[2] if eligible(k)]
        _validate_splits(info, (train, [], test), class_names, config.task,
                         config.candidate_policy, allow_empty_validation=True)
        external = {'train': train, 'validation': [], 'test': test, 'original_splits': original,
                    'split_config': split_metadata,
                    'excluded': [k for group in splits for k in group if not eligible(k)],
                    'samples': {k: {**info[k], 'labels': sorted(info[k]['labels']),
                                   'raw_path': str(Path(info[k]['raw_path']).resolve()),
                                   'chandra_path': str(Path(info[k]['chandra_path']).resolve()) if info[k].get('chandra_path') else None}
                                for k in train + test},
                    'source_signatures': {k: _source_signature(info[k]) for k in train + test}}
        _dump(root / 'external_split.json', external)
        from .progress import message
        message(f"CV interne : {cv.n_splits} folds ; train externe={len(train)}, test réservé={len(test)} ; résultats : {root}", config.verbose)
        plan = build_internal_folds(info, train, config)
        _dump(root / 'cv/folds.json', plan)
        audit = audit_initialization(config, info, train, test, plan['folds'])
        _dump(root / 'cv/initialization_audit.json', audit)
        if audit['external_test_leakage']:
            raise ValueError('; '.join(audit['warnings']))
        if audit['warnings']:
            if cv.source_provenance_policy == 'error':
                raise ValueError('; '.join(audit['warnings']))
            for problem in audit['warnings']:
                warnings.warn(problem, UserWarning)
        phases = _training_phases(config)
        if phases and cv.max_epochs is not None:
            raise ValueError('CV max_epochs is for single-phase training; use phase_durations for phases')
        if not phases and cv.phase_durations is not None:
            raise ValueError('CV phase_durations requires an existing training phase plan')
        candidates = cv.phase_durations if phases and cv.phase_durations else [
            {p.name: p.epochs for p in phases} if phases else None]
        if phases and any(set(c) != {p.name for p in phases} for c in candidates):
            raise ValueError('Each phase_durations candidate must specify every phase name exactly once')
        base = replace(config, cross_validation=None, use_early_stopping=False,
                       threshold_selection='fixed', epochs=cv.max_epochs or config.epochs)
        trials = []
        for trial_index, durations in enumerate(candidates):
            trial_config = _with_durations(base, durations) if phases else base
            histories, paths, sizes = [], [], []
            for fold in plan['folds']:
                _check_sources(audit)
                fold_index = fold['fold']
                fold_dir = root / 'cv' / f'trial_{trial_index}' / f'fold_{fold_index}'
                _dump(root / 'status.json', {'status': 'running', 'stage': 'cv', 'trial': trial_index, 'fold': fold_index})
                message(f"CV essai {trial_index + 1}/{len(candidates)}, fold {fold_index + 1}/{cv.n_splits}", config.verbose)
                # No test IDs, paths, datasets or loader are given to this training call.
                fold_info = {k: info[k] for k in fold['train'] + fold['validation']}
                run_experiment(fold_info, trial_config, root, class_names=class_names,
                    splits=(fold['train'], fold['validation'], []), model_factory=model_factory,
                    input_adapter=input_adapter,
                    _cv_execution={'stage': 'fold', 'run_dir': str(fold_dir),
                                   'evaluate_test': False, 'initialization_audit': audit})
                _check_sources(audit)
                histories.append([json.loads(line) for line in (fold_dir / 'history.jsonl').read_text().splitlines()])
                paths.append(str(fold_dir.relative_to(root)))
                sizes.append(len(fold['validation']))
            aggregation = aggregate_curves(histories, sizes, cv.selection_metric, cv.selection_mode)
            # A phase candidate is evaluated at its terminal state, never at independent phase minima.
            score = aggregation['curves'][-1]['mean'] if phases else aggregation['score']
            epochs = sum(durations.values()) if phases else aggregation['selected_epoch']
            if phases:
                aggregation.update(selected_epoch=epochs, score=score,
                                   tie_break='complete trajectory endpoint; no within-trial epoch selection')
            trials.append({'trial': trial_index, 'durations': durations, 'epochs': epochs,
                           'score': score, 'fold_paths': paths, 'aggregation': aggregation})
            _dump(root / 'cv/search_progress.json', {'completed_trials': trials})
        selected = min(trials, key=lambda t: (
            t['score'] if cv.selection_mode == 'min' else -t['score'], t['epochs'],
            tuple(t['durations'][p.name] for p in phases) if phases else (), t['trial']))
        selection = {'selected_trial': selected['trial'], 'selected_epoch': selected['epochs'],
                     'phase_durations': selected['durations'], 'metric': cv.selection_metric,
                     'mode': cv.selection_mode, 'score': selected['score'],
                     'rule': 'weighted terminal validation metric of complete phase trajectories' if phases else 'weighted mean validation curve optimum',
                     'tie_break': 'fewest total epochs, phase durations in configured order, trial index' if phases else 'earliest epoch',
                     'epoch_indexing': 'one-based', 'fixed_thresholds_during_search': config.thresholds,
                     'early_stopping': False}
        _dump(root / 'cv/epoch_selection.json', selection)
        message(f"CV : {selection['selected_epoch']} époques retenues ; {selection['metric']}={selection['score']:.6f}", config.verbose)
        predictions = []
        for fold_path in selected['fold_paths']:
            predictions.extend(_read(root / fold_path / 'validation_epochs' / f"epoch_{selected['epochs']}.json"))
        if len(predictions) != len(train) or {p['key'] for p in predictions} != set(train):
            raise ValueError('Incomplete out-of-fold predictions at selected epoch')
        from .metrics import select_multilabel_thresholds
        thresholds = None
        if config.task == 'rr_rh':
            thresholds = select_multilabel_thresholds(
                [p['label'] for p in predictions], [p['probabilities'] for p in predictions],
                target_mask=[p['label_mask'] for p in predictions], method=config.threshold_selection,
                grid=config.threshold_grid, min_precision=config.threshold_min_precision, fallback=config.thresholds)
            thresholds['selection_split'] = 'validation' if config.threshold_selection != 'fixed' else None
            thresholds['provenance'] = 'pooled internal out-of-fold predictions at selected epoch/phase combination'
            _dump(root / 'cv/thresholds.json', thresholds)
        _dump(root / 'cv/oof_predictions.json', predictions)
        summary = {'selection': selection, 'trials': trials, 'initialization_audit': audit,
                   'threshold_selection': thresholds,
                   'preprocessing': {'normalization': 'deterministic per-image percentiles 30/99 and asinh alpha=10; no population fit',
                                     'class_weights': 'computed only from each sub-train; recomputed on full train for final fit',
                                     'batchnorm': 'fresh model/source buffers per trial and fold; updated only in training',
                                     'cache': 'only deterministic per-image FITS/scattering transforms are shared'},
                   'refit_requested': cv.retrain_full_train}
        _dump(root / 'cv/cv_summary.json', summary)
        _plot_curves(root / 'cv/cv_curves.png', trials, bool(phases))
        if not cv.retrain_full_train:
            _dump(root / 'status.json', {'status': 'completed', 'stage': 'cv_only', 'test_evaluated': False})
            return root
        final_config = _with_durations(base, selected['durations']) if phases else replace(base, epochs=selected['epochs'])
        if thresholds:
            final_config = replace(final_config, thresholds=thresholds['thresholds'])
        _check_sources(audit)
        _dump(root / 'status.json', {'status': 'running', 'stage': 'final'})
        final = run_experiment({k: info[k] for k in train + test}, final_config, root,
            class_names=class_names, splits=(train, [], test), model_factory=model_factory, input_adapter=input_adapter,
            _cv_execution={'stage': 'final', 'run_dir': str(root / 'final'),
                           'evaluate_test': cv.evaluate_test_after_retrain,
                           'cross_validation': {'config': asdict(cv), 'selection': selection,
                               'initialization_audit': audit, 'threshold_selection': thresholds,
                               'summary_path': str((root / 'cv/cv_summary.json').resolve())}})
        _check_sources(audit)
        _dump(root / 'status.json', {'status': 'completed', 'stage': 'final',
                                   'reference_model': 'final/best.pt', 'test_evaluated': cv.evaluate_test_after_retrain})
        return final
    except Exception as error:
        state = _read(root / 'status.json')
        _dump(root / 'status.json', {**state, 'status': 'failed', 'error': str(error)})
        raise
