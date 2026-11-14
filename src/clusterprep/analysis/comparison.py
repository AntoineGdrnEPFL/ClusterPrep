"""Metric definitions and conservative, explicit comparison contracts."""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score

from ..metrics import classification_metrics, multilabel_metrics
from .registry import fingerprint, flatten, prediction_table


def label_contract(run):
    return [run.task_type, run.config.get('task'), run.labels,
            run.config.get('candidate_policy'), run.metrics.get('annotation_convention')]


def partition(run):
    samples = run.splits.get('samples', {})
    if not samples or not all(s in run.splits for s in ('train', 'validation', 'test')):
        return None
    result = {}
    for split in ('train', 'validation', 'test'):
        try:
            result[split] = sorted(set(str(samples[k]['cluster_id']) for k in run.splits[split]))
        except KeyError:
            return None
    return result


def protocol_id(run, split, cohort='all', threshold_dependent=True):
    """Fold protocols may aggregate; missing provenance isolates a run."""
    parts = partition(run)
    if parts is None or run.status == 'invalid':
        return 'unverified-' + run.run_id
    split_config = {k: v for k, v in run.splits.get('split_config', {}).items() if k != 'fold_index'}
    # Historical manifests omitted use_val; infer it from the recorded partition,
    # not from a blanket default. Preserve explicit policies and training provenance.
    split_config.setdefault('use_val', bool(parts['validation']))
    if run.fold is not None and (run.cv_plan or split_config.get('n_folds')):
        dataset = sorted(set().union(*map(set, parts.values())))
        plan = run.cv_plan or split_config
    else:
        dataset, plan = parts, None
    evaluation = None
    if run.fold is None:
        evaluated = prediction_table([run], split, cohort)
        if not evaluated.empty:
            evaluation = evaluated.sort_values(['cluster_id', 'label'])[['cluster_id', 'label', 'truth', 'known']].to_dict('records')
        else:
            # Scores without evaluated labels are descriptive, not evidence of comparability.
            evaluation = ['unverified', run.run_id]
    annotations = sorted((str(v.get('cluster_id')), v.get('labels')) for v in run.splits.get('samples', {}).values())
    policy = run.threshold_policy if threshold_dependent else None
    if threshold_dependent and run.task_type == 'multilabel' and run.thresholds is None:
        policy = ['unknown', run.run_id]
    return fingerprint([label_contract(run), dataset, annotations, plan, run.stage, split, cohort, policy, evaluation])


def assert_comparable(runs, split='test', cohort='all', threshold_dependent=True):
    """Pairing requires all three partitions AND evaluated truths/masks to match."""
    if len(runs) < 2:
        return
    first = runs[0]
    parts = partition(first)
    if parts is None or any(r.status == 'invalid' for r in runs):
        raise ValueError('Split provenance missing or invalid')
    for run in runs[1:]:
        if label_contract(run) != label_contract(first):
            raise ValueError('Incompatible tasks or label definitions')
        if run.stage != first.stage or partition(run) != parts:
            raise ValueError('Unpaired partitions or evaluation stages')
        if threshold_dependent and (run.threshold_policy != first.threshold_policy or
                (run.task_type == 'multilabel' and (run.thresholds is None or first.thresholds is None))):
            raise ValueError('Incompatible or unknown threshold policies')
    signatures = []
    for run in runs:
        table = prediction_table([run], split, cohort)
        if table.empty:
            raise ValueError('Detailed predictions required for pairing')
        signatures.append(table.sort_values(['cluster_id', 'label'])[['cluster_id', 'label', 'truth', 'known']].reset_index(drop=True))
    if any(not signatures[0].equals(t) for t in signatures[1:]):
        raise ValueError('Incompatible evaluated clusters, truths or annotation masks')


def score_run(run, split='test', cohort='all'):
    frame = run.predictions.get(split, pd.DataFrame())
    stored = flatten(run.metrics.get(split) or {}) if cohort == 'all' else {}
    if frame.empty:
        if run.task_type == 'multilabel' and run.thresholds is None:
            stored = {k: v for k, v in stored.items() if any(n in k for n in ('roc_auc', 'average_precision', 'loss', 'brier'))}
        return stored, 'stored'
    if cohort != 'all':
        frame = frame[frame.has_chandra if cohort == 'paired' else ~frame.has_chandra]
    if frame.empty:
        return {}, 'recomputed'
    labels = run.labels
    p = frame[[f'probability_{n}' for n in labels]].to_numpy(float)
    if run.task_type == 'binary':
        result = classification_metrics(frame.label.to_numpy(), p, labels)
        for metric in ('precision', 'recall'):
            result['macro_' + metric] = float(np.mean([result['per_class'][n][metric] for n in labels]))
            result['micro_' + metric] = result['accuracy']
    else:
        y = frame[[f'label_{n}' for n in labels]].to_numpy(float)
        mask = frame[[f'known_{n}' for n in labels]].to_numpy(bool)
        if run.thresholds is None:
            result = {}
            for j, label in enumerate(labels):
                valid = mask[:, j]
                if len(np.unique(y[valid, j])) == 2:
                    result[f'per_label.{label}.roc_auc'] = roc_auc_score(y[valid, j], p[valid, j])
                    result[f'per_label.{label}.average_precision'] = average_precision_score(y[valid, j], p[valid, j])
            for metric in ('roc_auc', 'average_precision'):
                values = [value for key, value in result.items() if key.endswith('.' + metric)]
                result[metric + '_macro'] = float(np.mean(values)) if values else None
                result[metric + '_macro_n_labels'] = len(values)
            # Do not expose historical threshold-dependent metrics with unknown cutoffs.
            return result, 'continuous_only'
        result = multilabel_metrics(y, p, target_mask=mask, thresholds=run.thresholds)
        predicted = p >= np.array([run.thresholds[n] for n in labels])
        for name, function in [('precision', precision_score), ('recall', recall_score)]:
            result['micro_' + name] = float(function(y[mask], predicted[mask], zero_division=0)) if mask.any() else None
            values = [result['per_label'][n][name] for n in labels if result['per_label'][n][name] is not None]
            result['macro_' + name] = float(np.mean(values)) if values else None
    # Preserve saved loss (possibly weighted), do not substitute unweighted log-loss.
    return {**stored, **flatten(result)}, 'recomputed'


def metrics_table(runs, split='test', cohort='all'):
    if split not in ('train', 'validation', 'test') or cohort not in ('all', 'paired', 'raw_only'):
        raise ValueError('Unsupported split/cohort')
    rows = []
    for run in runs:
        metrics, source = score_run(run, split, cohort)
        row = run.summary()
        row.update(split=split, cohort=cohort, metric_source=source,
                   protocol_id=protocol_id(run, split, cohort))
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_metrics(table, metrics=('macro_f1', 'roc_auc_macro')):
    """One vote per fold; repetitions are averaged within fold before dispersion."""
    rows = []
    if table.empty:
        return pd.DataFrame()
    complete = table[table.status == 'completed']
    group_keys = ['configuration_id', 'protocol_id', 'task_type', 'task', 'model', 'modalities', 'noise_method', 'split', 'cohort', 'metric_source']
    for keys, group in complete.groupby(group_keys, dropna=False):
        row = dict(zip(group_keys, keys))
        expected = group.expected_folds.dropna()
        row['radio_source'] = group.radio_source.iloc[0] if 'radio_source' in group else 'RAW'
        row.update(configuration_name='; '.join(sorted(set(group.configuration_name))),
                   run_names='; '.join(group.run_name), n_runs=len(group), n_folds=group.fold.nunique(),
                   expected_folds=expected.max() if len(expected) else None,
                   available_folds=','.join(str(int(f)) for f in sorted(group.fold.dropna().unique())),
                   experiment_ids='; '.join(sorted(set(group.experiment_id))),
                   run_ids='; '.join(group.run_id))
        # Runs without folds are repetitions, never reported as folds.
        unit = group.fold.astype(object).where(group.fold.notna(), group.run_id)
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors='coerce').groupby(unit).mean().dropna()
            row.update({metric + '_mean': values.mean(), metric + '_std': values.std(ddof=1),
                        metric + '_n': len(values)})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_history(epochs, metric='validation.loss'):
    """Available observations only, grouped by phase-local epoch; no extrapolation."""
    if epochs.empty or metric not in epochs:
        return pd.DataFrame()
    epochs = epochs.copy()
    epochs['unit'] = epochs.fold.astype(object).where(epochs.fold.notna(), epochs.run_id)
    keys = ['configuration_id', 'protocol_id', 'phase', 'phase_epoch']
    per_fold = epochs.groupby(keys + ['unit'])[metric].mean().reset_index()
    result = per_fold.groupby(keys)[metric].agg(['mean', 'std', 'count']).reset_index()
    names = epochs.groupby('configuration_id').configuration_name.agg(lambda values: '; '.join(sorted(set(values))))
    result['configuration_name'] = result.configuration_id.map(names)
    return result


def paired_comparisons(runs, split='test', cohort='all', metric='macro_f1'):
    rows = []
    for left, right in combinations(runs, 2):
        row = dict(left_name=left.run_name, right_name=right.run_name, left=left.run_id, right=right.run_id, fold_left=left.fold, fold_right=right.fold,
                   metric=metric, split=split, paired=False, reason='', delta=np.nan)
        try:
            assert_comparable([left, right], split, cohort,
                              threshold_dependent=not any(s in metric for s in ('roc_auc', 'average_precision', 'loss', 'brier')))
            a, _ = score_run(left, split, cohort)
            b, _ = score_run(right, split, cohort)
            if a.get(metric) is None or b.get(metric) is None:
                raise ValueError('Metric unavailable')
            row.update(paired=True, delta=b[metric] - a[metric])
        except ValueError as exc:
            row['reason'] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_predictions(runs, split='test', cohort='all'):
    """Align on cluster IDs; never pool repeated fixed-test predictions as OOF."""
    if len(runs) < 2:
        return pd.DataFrame()
    assert_comparable(runs, split, cohort)
    frame = prediction_table(runs, split, cohort)
    if frame.duplicated(['run_id', 'cluster_id', 'label']).any():
        raise ValueError('Duplicate predictions')
    rows = []
    for (cluster, label), group in frame.groupby(['cluster_id', 'label']):
        known = group.known.all() and group.correct.notna().all()
        rows.append(dict(cluster_id=cluster, label=label, truth=group.truth.iloc[0],
                         all_correct=bool(group.correct.all()) if known else None,
                         all_incorrect=bool((~group.correct.astype(bool)).all()) if known else None,
                         disagreement=bool(group.correct.nunique() > 1) if known else None,
                         probability_range=group.score.max() - group.score.min(),
                         correct_runs='; '.join(group.loc[group.correct == True, 'run_name']),
                         incorrect_runs='; '.join(group.loc[group.correct == False, 'run_name'])))
    return pd.DataFrame(rows).sort_values('probability_range', ascending=False)


def transfer_table(runs):
    rows = []
    for run in runs:
        transfer = run.config.get('transfer') or {}
        phases = run.phase_report.get('phases') or run.config.get('training_phases') or transfer.get('phases') or [{}]
        for phase in phases:
            rows.append(dict(run_name=run.run_name, run_id=run.run_id, configuration_name=run.configuration_name, configuration_id=run.configuration_id,
                             strategy=run.transfer_strategy, source=transfer,
                             target_task=run.config.get('task'), phase=phase.get('name', 'training'),
                             phase_config=phase, selected_phase=run.phase_report.get('selected_phase')))
    return pd.DataFrame(rows)


def out_of_fold_predictions(runs, cohort='all'):
    """Pool internal validation only after proving disjointness and full coverage."""
    if not runs or any(r.stage != 'fold' or r.status != 'completed' for r in runs):
        raise ValueError('OOF requires completed internal CV folds, never a fixed test set')
    if len({r.configuration_id for r in runs}) != 1 or len({protocol_id(r, 'validation', cohort) for r in runs}) != 1:
        raise ValueError('OOF requires one configuration and one compatible CV protocol')
    if len({r.metrics.get('best_epoch') for r in runs}) != 1:
        raise ValueError('OOF predictions must use the same selected epoch across folds')
    frame = prediction_table(runs, 'validation', 'all')
    if frame.empty or frame.duplicated(['cluster_id', 'label']).any():
        raise ValueError('OOF coverage missing or overlapping validation clusters')
    parts = [partition(r) for r in runs]
    if any(p is None for p in parts):
        raise ValueError('OOF manifest missing')
    expected = set(parts[0]['train']) | set(parts[0]['validation'])
    if any((set(p['train']) | set(p['validation'])) != expected for p in parts):
        raise ValueError('OOF training universes differ')
    if set(frame.cluster_id) != expected:
        raise ValueError('OOF folds incomplete: validation does not cover the training universe')
    frame['prediction_role'] = 'out_of_fold'
    if cohort != 'all':
        if cohort not in ('paired', 'raw_only'):
            raise ValueError('Unknown cohort')
        frame = frame[frame.has_chandra if cohort == 'paired' else ~frame.has_chandra].copy()
    frame['cohort'] = cohort
    return frame


def metrics_by_fold(table, metrics=('macro_f1', 'roc_auc_macro')):
    """One row per configuration/protocol/fold, with repetition counts."""
    if table.empty:
        return pd.DataFrame()
    rows = []
    for fold, group in table.groupby('fold', dropna=False):
        summary = aggregate_metrics(group, metrics)
        if not summary.empty:
            summary['fold'] = fold
            rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
