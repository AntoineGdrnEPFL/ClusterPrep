"""Read-only discovery of training artifacts, independent of model/checkpoint code."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

from ..labels import task_target
from ..metrics import resolve_thresholds


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:20]


def read_json(path):
    def invalid(value):
        raise ValueError(f'Non-finite JSON: {value}')
    value = json.loads(Path(path).read_text(), parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif item is None or isinstance(item, (int, float, bool)):
            result[name] = item
    return result


@dataclass
class TrainingRun:
    path: Path
    metadata: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    splits: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    predictions: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    status: str = 'incomplete'
    thresholds: dict | None = None
    threshold_policy: dict = field(default_factory=dict)
    cv_plan: dict = field(default_factory=dict)
    phase_report: dict = field(default_factory=dict)
    fold: int | None = None

    @property
    def run_id(self):
        return fingerprint(str(self.path))

    @property
    def run_name(self):
        """Folder name without the pipeline's timestamp/random suffix."""
        return re.sub(r'-\d{8}T\d{6}-[0-9a-fA-F]{8}$', '', self.path.name)

    @property
    def figure_stem(self):
        """Readable export name, unique even for repeated runs with the same name."""
        return f'{self.run_name}__{self.run_id}'

    @property
    def diagnostic_labels(self):
        if self.task_type != 'binary':
            return self.labels
        target = {'de_nde': 'DE', 'rr_binary': 'RR', 'rh_binary': 'RH'}.get(self.config.get('task'))
        if target not in self.labels:
            target = 'DE' if 'DE' in self.labels else self.labels[-1]
        return [target]

    @property
    def configuration_name(self):
        # Aggregate labels omit only the verified fold suffix, keeping the model,
        # classification and augmentation names from the folder.
        return (re.sub(rf'_fold_{self.fold}$', '', self.run_name)
                if self.fold is not None else self.run_name)

    @property
    def task_type(self):
        if self.config.get('task') == 'rr_rh':
            return 'multilabel'
        if len(self.labels) == 2:
            return 'binary'
        return 'unknown'

    @property
    def labels(self):
        if self.config.get('task') == 'rr_rh':
            return ['RR', 'RH']
        return self.metrics.get('class_names', self.metadata.get('class_names', []))

    @property
    def stage(self):
        return self.metadata.get('execution', {}).get('stage', 'standard')

    @property
    def experiment_id(self):
        return self.config.get('name') or 'unknown'

    @property
    def configuration_id(self):
        # Seed stays in the configuration: folds of a seeded protocol may aggregate,
        # independent seeds must not silently count as extra folds.
        ignored = {'name', 'verbose', 'device', 'num_workers', 'cache'}
        config = {k: v for k, v in self.config.items() if k not in ignored}
        split = {k: v for k, v in self.splits.get('split_config', {}).items() if k != 'fold_index'}
        return fingerprint([config, split, self.stage, self.metadata.get('normalization'),
                            self.metadata.get('library_source_sha256')])

    @property
    def noise_method(self):
        noise = (self.config.get('augmentation') or {}).get('radio_noise')
        if noise is None:
            return 'unspecified'
        return noise.get('method', 'unknown') if noise.get('enabled') else 'disabled'

    @property
    def transfer_strategy(self):
        return 'transfer' if self.config.get('transfer') else 'scratch'

    def summary(self):
        return dict(run_name=self.run_name, run_id=self.run_id, configuration_name=self.configuration_name, path=str(self.path), experiment_id=self.experiment_id,
                    configuration_id=self.configuration_id, model=self.config.get('model'),
                    architecture=self.metadata.get('model_class', self.config.get('model')),
                    modalities=self.config.get('mode'), task_type=self.task_type,
                    task=self.config.get('task'), noise_method=self.noise_method, labels=json.dumps(self.labels), fold=self.fold,
                    expected_folds=len(self.cv_plan.get('folds', [])) or self.splits.get('split_config', {}).get('n_folds'),
                    seed=self.config.get('seed'), stage=self.stage, status=self.status,
                    transfer_strategy=self.transfer_strategy, best_epoch=self.metrics.get('best_epoch'),
                    epochs=self.metrics.get('epochs_ran', len(self.history) or None),
                    stopped_early=self.metrics.get('stopped_early'), seconds=self.metrics.get('seconds'),
                    thresholds=json.dumps(self.thresholds), threshold_policy=json.dumps(self.threshold_policy),
                    config=json.dumps(self.config, sort_keys=True),
                    checkpoint_path=str(self.path/'best.pt') if (self.path/'best.pt').exists() else None,
                    artifacts=json.dumps({p.name: str(p) for p in self.path.iterdir() if p.is_file()}),
                    issues='; '.join(self.issues))


def _load_predictions(run, split):
    path = run.path / f'{split}_predictions.csv'
    epoch_path = run.path / 'validation_epochs' / f"epoch_{run.metrics.get('best_epoch')}.json"
    if not path.exists() and not (split == 'validation' and epoch_path.exists()):
        if run.splits.get(split):
            run.issues.append(f'{split}: detailed predictions missing')
        return pd.DataFrame()
    names = run.labels
    if path.exists():
        frame = pd.read_csv(path, dtype={'key': str, 'cluster_id': str})
    else:
        records = json.loads(epoch_path.read_text())
        normalized = []
        for record in records:
            row = {k: record[k] for k in ('key', 'cluster_id', 'has_chandra')}
            for i, name in enumerate(names):
                row[f'probability_{name}'] = record['probabilities'][i]
                if run.task_type == 'multilabel':
                    row[f'label_{name}'] = record['label'][i]
                    row[f'known_{name}'] = record['label_mask'][i]
                    row[f'prediction_{name}'] = record['prediction'][i]
            if run.task_type == 'binary':
                row.update(label=record['label'], prediction=record['prediction'])
            normalized.append(row)
        frame = pd.DataFrame(normalized)
        run.issues.append(f'validation: loaded epoch {run.metrics.get("best_epoch")} predictions (run best/last, not pooled OOF)')
    required = {'key', 'cluster_id', 'has_chandra'} | {f'probability_{n}' for n in names}
    required |= ({'label', 'prediction'} if run.task_type == 'binary' else {f'label_{n}' for n in names})
    if not names or not required <= set(frame):
        raise ValueError(f'{split}: missing columns {required-set(frame)} / labels')
    for col in ('key', 'cluster_id'):
        if frame[col].isna().any() or frame[col].duplicated().any():
            raise ValueError(f'{split}: missing/duplicate {col}')
    from ..multilabel_analysis import _boolean
    frame['has_chandra'] = frame.has_chandra.map(_boolean)
    probabilities = frame[[f'probability_{n}' for n in names]].to_numpy(float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f'{split}: invalid probabilities')
    if split in run.splits and set(frame.key) != set(run.splits[split]):
        raise ValueError(f'{split}: prediction keys differ from manifest')
    samples = run.splits.get('samples', {})
    if samples:
        for row in frame.itertuples():
            if str(samples.get(row.key, {}).get('cluster_id')) != row.cluster_id:
                raise ValueError(f'{split}: cluster identity differs from manifest')
    if run.task_type == 'binary':
        if not np.allclose(probabilities.sum(1), 1, atol=1e-5):
            raise ValueError(f'{split}: binary probabilities must sum to one')
        if not frame.label.isin(range(len(names))).all() or not np.array_equal(frame.prediction, probabilities.argmax(1)):
            raise ValueError(f'{split}: invalid label or non-argmax prediction')
        for record in frame.to_dict('records'):
            sample = samples.get(record['key'], {})
            if sample.get('labels') and run.config.get('task') in ('de_nde', 'rr_binary', 'rh_binary'):
                expected, _ = task_target(sample, run.config['task'], run.config.get('candidate_policy', 'positive'), names)
                if record['label'] != expected:
                    raise ValueError(f'{split}: binary truth differs from manifest')
        for i, name in enumerate(names):
            frame[f'label_{name}'] = frame.label.eq(i).astype(int)
            frame[f'prediction_{name}'] = frame.prediction.eq(i).astype(int)
            frame[f'known_{name}'] = True
    else:
        known_fields = {f'known_{n}' for n in names}
        if set(frame) & known_fields and not known_fields <= set(frame):
            raise ValueError(f'{split}: partial annotation mask columns')
        # Reuse the existing schema/annotation validator when its prerequisites exist.
        if path.exists() and samples and run.thresholds is not None:
            from ..multilabel_analysis import _read_predictions
            p = _read_predictions(path, split, run.splits, run.config, run.thresholds, run.issues)
            for i, name in enumerate(names):
                frame[f'known_{name}'] = p.mask[:, i]
        for name in names:
            known = f'known_{name}'
            if known not in frame:
                if run.config.get('candidate_policy') == 'positive':
                    frame[known] = True
                elif samples and all('labels' in samples.get(k, {}) for k in frame.key):
                    i = names.index(name)
                    frame[known] = [bool(task_target(samples[k], 'rr_rh', run.config.get('candidate_policy'))[1][i]) for k in frame.key]
                else:
                    raise ValueError(f'{split}: unknown annotation mask')
            frame[known] = frame[known].map(_boolean)
            if not frame.loc[frame[known], f'label_{name}'].isin([0, 1]).all():
                raise ValueError(f'{split}: invalid known labels')
            if samples and run.config.get('candidate_policy') in ('positive', 'ignore'):
                j = names.index(name)
                for record in frame.to_dict('records'):
                    sample = samples.get(record['key'], {})
                    if 'labels' in sample:
                        expected, mask = task_target(sample, 'rr_rh', run.config['candidate_policy'])
                        if record[known] and (not mask[j] or record[f'label_{name}'] != expected[j]):
                            raise ValueError(f'{split}: labels/masks differ from manifest')
            if run.thresholds is not None and f'prediction_{name}' in frame:
                if not np.array_equal(frame[f'prediction_{name}'], frame[f'probability_{name}'].ge(run.thresholds[name])):
                    raise ValueError(f'{split}: saved predictions differ from thresholds')
            frame[f'prediction_{name}'] = (frame[f'probability_{name}'].ge(run.thresholds[name]).astype(int)
                                           if run.thresholds is not None else np.nan)
    return frame


def load_run(directory):
    run = TrainingRun(Path(directory).resolve())
    artifacts = {}
    for name in ('config', 'metrics', 'splits', 'status', 'thresholds', 'training_phases'):
        path = run.path / f'{name}.json'
        try:
            artifacts[name] = read_json(path) if path.exists() else {}
        except (OSError, ValueError) as exc:
            artifacts[name] = {}
            run.issues.append(f'{name}.json: {exc}')
    run.metadata, run.metrics, run.splits = (artifacts[n] for n in ('config', 'metrics', 'splits'))
    run.phase_report = artifacts['training_phases']
    run.config = {**run.metadata.get('config', run.metadata), **run.metrics.get('config', {})}
    if not run.config.get('model'):
        raise ValueError('No model configuration (not a training run)')
    if not any((run.path / name).exists() for name in ('metrics.json', 'history.jsonl', 'history.csv', 'status.json')) and not any(k in run.metadata for k in ('class_names', 'model_class', 'created_at')):
        raise ValueError('No training artifacts (configuration only)')
    declared = artifacts['status'].get('status')
    # Legacy artifacts lack a completion marker: show metrics, but never assert completion.
    run.status = declared or 'unverified'
    if declared == 'completed' and not run.metrics:
        run.status = 'incomplete'
        run.issues.append('Completion marker without valid metrics')
    if not declared:
        run.issues.append('Completion status unavailable')
    for key in ('task', 'name', 'mode', 'seed'):
        if key not in run.config:
            run.issues.append(f'Missing metadata: {key}')
    if not run.splits:
        run.issues.append('Split manifest missing; comparisons cannot be verified')
    if not (run.path/'best.pt').exists():
        run.issues.append('Checkpoint missing (not required for analysis)')
    selected = artifacts['thresholds'] or run.metrics.get('threshold_selection') or run.metadata.get('evaluation', {})
    if run.task_type == 'multilabel':
        values = selected.get('thresholds', run.config.get('thresholds'))
        if values is not None:
            run.thresholds = resolve_thresholds(values)
        else:
            run.issues.append('Thresholds unknown: threshold-dependent scores disabled')
        run.threshold_policy = {k: selected.get(k) for k in ('method', 'selection_split', 'provenance')}
        if not run.threshold_policy.get('method') and values is not None:
            run.threshold_policy['method'] = run.config.get('threshold_selection', 'fixed')
        if run.threshold_policy.get('method') == 'fixed':
            run.threshold_policy['thresholds'] = run.thresholds
        elif run.threshold_policy.get('selection_split') != 'validation':
            run.issues.append('Threshold selection provenance unknown or not validation')
            run.thresholds = None
    else:
        run.threshold_policy = {'method': 'argmax', 'class_order': run.labels}
    split_config = run.splits.get('split_config', {})
    run.fold = split_config.get('fold_index') if split_config.get('n_folds') else None
    if run.stage == 'fold':
        # Current CV artifacts omit fold_index; verified pipeline layout fallback.
        match = re.fullmatch(r'fold_(\d+)', run.path.name)
        if match:
            run.fold = int(match[1])
            run.issues.append('Fold inferred from pipeline layout (execution.stage=fold)')
        plan_path = run.path.parent.parent/'folds.json'
        if plan_path.exists():
            try:
                run.cv_plan = read_json(plan_path)
            except ValueError as exc:
                run.issues.append(f'CV plan: {exc}')
    groups = []
    for split in ('train', 'validation', 'test'):
        samples = run.splits.get('samples', {})
        ids = [str(samples.get(k, {}).get('cluster_id', k)) for k in run.splits.get(split, [])]
        if len(ids) != len(set(ids)):
            run.issues.append(f'{split}: duplicate clusters in manifest')
        groups.append(set(ids))
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        run.issues.append('Invalid split manifest: cluster leakage')
        run.status = 'invalid'
    history = run.path/'history.jsonl'
    if history.exists():
        for line_number, line in enumerate(history.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or 'epoch' not in row:
                    raise ValueError('missing epoch')
                run.history.append(row)
            except ValueError as exc:
                run.issues.append(f'history line {line_number}: {exc}')
    elif (run.path/'history.csv').exists():
        try:
            run.history = pd.read_csv(run.path/'history.csv').to_dict('records')
            if any('epoch' not in row for row in run.history):
                raise ValueError('missing epoch')
        except (ValueError, OSError) as exc:
            run.history = []
            run.issues.append(f'history.csv: {exc}')
    else:
        run.issues.append('History missing')
    for split in ('validation', 'test'):
        try:
            frame = _load_predictions(run, split)
            if not frame.empty:
                run.predictions[split] = frame
        except (ValueError, KeyError, TypeError, OSError) as exc:
            run.issues.append(f'{split} predictions: {exc}')
    return run


def discover_runs(root, recursive=True):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return [], pd.DataFrame([{'path': str(root), 'issue': 'Root directory missing'}])
    paths = set()
    for name in ('config.json', 'metrics.json', 'status.json'):
        paths.update(p.parent.resolve() for p in (root.rglob(name) if recursive else root.glob(name)))
    runs, issues = [], []
    for path in sorted(paths):
        # Phase metrics and CV orchestration status are not independent trained models.
        if not (path/'config.json').exists() and not (path/'metrics.json').exists():
            continue
        try:
            run = load_run(path)
            runs.append(run)
            issues.extend({'path': str(path), 'issue': issue} for issue in run.issues)
        except (ValueError, TypeError, AttributeError, OSError, KeyError) as exc:
            if str(exc).startswith('No model configuration') and not (path/'config.json').exists():
                continue
            issues.append({'path': str(path), 'issue': str(exc)})
    return runs, pd.DataFrame(issues, columns=['path', 'issue'])


def filter_runs(runs, **filters):
    """None means no filtering; scalar or iterable values are accepted."""
    selected = []
    for run in runs:
        row = run.summary()
        if all(value is None or row.get(key) in (value if isinstance(value, (list, tuple, set)) else [value])
               for key, value in filters.items()):
            selected.append(run)
    return selected


def registry_table(runs):
    return pd.DataFrame([r.summary() for r in runs])


def epoch_table(runs):
    rows = []
    from .comparison import protocol_id
    for run in runs:
        counts = {}
        for record in run.history:
            phase = record.get('phase', 'training')
            counts[phase] = counts.get(phase, 0) + 1
            row = dict(run_name=run.run_name, run_id=run.run_id, configuration_name=run.configuration_name, configuration_id=run.configuration_id,
                       protocol_id=protocol_id(run, 'validation'), fold=run.fold, phase=phase, epoch=record['epoch'], phase_epoch=record.get('phase_epoch', counts[phase]))
            row.update(flatten(record))
            for key in list(row):
                if key.startswith('val_'):
                    row['validation.' + key[4:]] = row.pop(key)
                elif key.startswith('train_'):
                    row['train.' + key[6:]] = row.pop(key)
            rows.append(row)
    return pd.DataFrame(rows)


def prediction_table(runs, split='test', cohort='all'):
    rows = []
    for run in runs:
        frame = run.predictions.get(split, pd.DataFrame())
        if cohort not in ('all', 'paired', 'raw_only'):
            raise ValueError('Unknown cohort')
        if not frame.empty and cohort != 'all':
            frame = frame[frame.has_chandra if cohort == 'paired' else ~frame.has_chandra]
        for record in frame.to_dict('records'):
            for label in run.labels:
                known = bool(record[f'known_{label}'])
                truth = record[f'label_{label}'] if known else np.nan
                pred = record[f'prediction_{label}']
                rows.append(dict(run_name=run.run_name, run_id=run.run_id, configuration_name=run.configuration_name, configuration_id=run.configuration_id,
                                 model=run.config.get('model'), fold=run.fold, split=split, cohort=cohort,
                                 prediction_role='internal_validation' if run.stage == 'fold' else 'held_out',
                                 key=record['key'], cluster_id=record['cluster_id'], label=label,
                                 truth=truth, known=known, score=record[f'probability_{label}'], prediction=pred,
                                 correct=truth == pred if known and pd.notna(pred) else None,
                                 has_chandra=record['has_chandra']))
    return pd.DataFrame(rows)
