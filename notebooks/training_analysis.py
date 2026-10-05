"""Analyse hors ligne des artefacts clusterprep, sans charger les checkpoints."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve

METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1',
           'roc_auc_macro', 'average_precision_macro', 'log_loss', 'brier']
LOWER = {'loss', 'log_loss', 'brier'}
GAIN_METRICS = ['accuracy', 'balanced_accuracy', 'f1_de', 'recall_de',
                'precision_de', 'macro_f1', 'roc_auc', 'pr_auc']


def gain_scores(frame, classes):
    """Scores de la heatmap ; ROC/AP ont DE pour classe positive."""
    if 'DE' not in classes:
        raise ValueError('DE class missing: DE metrics are undefined')
    result = scores(frame, classes)
    i = classes.index('DE')
    y = frame.label.to_numpy(int)
    p = frame[[f'probability_{c}' for c in classes]].to_numpy(float)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, p.argmax(1), labels=range(len(classes)), zero_division=0)
    binary = y == i
    both = 0 < binary.sum() < len(y)
    result.update(f1_de=f1[i] if support[i] else np.nan,
                  recall_de=recall[i] if support[i] else np.nan,
                  precision_de=precision[i] if support[i] else np.nan,
                  roc_auc=roc_auc_score(binary, p[:, i]) if both else np.nan,
                  pr_auc=average_precision_score(binary, p[:, i]) if both else np.nan)
    return {k: result[k] for k in GAIN_METRICS}


def gain_tables(runs, split='test', cohort='paired', include_pair=True,
                allow_code_change_models=()):
    """Apparie les mêmes modèles/protocoles ; pair/raw autorise le filtrage Chandra.

    Les splits de base doivent être égaux : aucun appariement par timestamp ou score.
    Une comparaison pair/raw filtrée reste descriptive (population train différente).
    """
    lookup = {r.id: r for r in runs}
    catalog = chandra_pairs(runs)
    rows, issues = [], []
    for index, pair in catalog.iterrows():
        if pd.isna(pair.raw_id):
            issues.append({'pair_index': index, **pair.to_dict()})
            continue
        raw, chandra = lookup[pair.raw_id], lookup[pair.chandra_id]
        if chandra.config.get('mode') == 'pair' and not include_pair:
            issues.append({'pair_index': index, **pair.to_dict(), 'error': 'pair/raw disabled'})
            continue
        reasons = pair_audit(raw, chandra)
        code_changed = bool(
            pair.model in allow_code_change_models
            and raw.metadata.get('library_source_sha256')
            and chandra.metadata.get('library_source_sha256')
            and 'metadata.library_source_sha256' in reasons)
        blocking_reasons = [r for r in reasons
                            if not (code_changed and r == 'metadata.library_source_sha256')]
        filtered_pair = False
        if blocking_reasons and include_pair and chandra.config.get('mode') == 'pair':
            base_a = raw.splits.get('base_splits')
            base_b = chandra.splits.get('base_splits')
            same_base = bool(base_a and base_b) and all(
                s in base_a and s in base_b and sorted(base_a[s]) == sorted(base_b[s])
                for s in ['train', 'validation', 'test'])
            samples = chandra.splits.get('samples', {})
            expected_filter = same_base and all(
                all(k in samples for k in base_a[s])
                and sorted(raw.splits.get(s, [])) == sorted(base_a[s])
                and sorted(chandra.splits.get(s, [])) == sorted(
                    k for k in base_a[s] if samples[k].get('chandra_path'))
                for s in ['train', 'validation', 'test'])
            filtered_pair = expected_filter and all(r.startswith('split.') for r in blocking_reasons)
        if blocking_reasons and not filtered_pair:
            issues.append({'pair_index': index, **pair.to_dict()})
            continue
        try:
            a, b = paired_frames(raw, chandra, split, cohort)
            ma, mb = gain_scores(a, raw.classes), gain_scores(b, chandra.classes)
            row = dict(pair_index=index, model=pair.model,
                       comparison=f"{chandra.config['mode']}/raw", raw_id=raw.id,
                       chandra_id=chandra.id, n=len(a), n_clusters=a.cluster_id.nunique(),
                       split=split, cohort=cohort, controlled=not reasons,
                       code_changed=code_changed, training_population_changed=filtered_pair,
                       raw_code_sha256=raw.metadata.get('library_source_sha256'),
                       chandra_code_sha256=chandra.metadata.get('library_source_sha256'),
                       raw_parameters=raw.metadata.get('parameters'),
                       chandra_parameters=chandra.metadata.get('parameters'),
                       differences=', '.join(reasons),
                       label=f"{pair.model} | {chandra.config['mode']}/raw | p{index} | n={len(a)}"
                             + (' *' if filtered_pair else '')
                             + (' [code change]' if code_changed else ''))
            for k in GAIN_METRICS:
                row[k] = mb[k] - ma[k]
                row[f'raw_{k}'], row[f'chandra_{k}'] = ma[k], mb[k]
            rows.append(row)
        except (ValueError, OSError) as exc:
            issues.append({'pair_index': index, **pair.to_dict(), 'error': str(exc)})
    details = pd.DataFrame(rows)
    if details.empty:
        matrix = pd.DataFrame(columns=GAIN_METRICS)
    else:
        details = details.sort_values(['model', 'comparison', 'chandra_id', 'raw_id'])
        matrix = details.set_index('label')[GAIN_METRICS]
    return matrix, details, pd.DataFrame(issues)


def plot_gain_heatmap(matrix, title='X-ray gain per run (blue = positive)',
                      annotate=True, limit=None):
    """Échelle divergente symétrique, valeurs absentes en gris, aucune imputation."""
    if matrix.empty:
        return None
    values = matrix.to_numpy(float)
    finite = values[np.isfinite(values)]
    vmax = max(float(np.abs(finite).max()) if finite.size else 0, .01) if limit is None else float(limit)
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError('Color limit must be finite and strictly positive')
    fig, ax = plt.subplots(figsize=(max(3.8, .4 * len(matrix) + 2), max(3.8, .38 * len(matrix) + 2)), layout='constrained')
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad('#d9d9d9')
    artist = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-vmax, vmax=vmax,
                       aspect='auto', interpolation='nearest')
    ax.set(xticks=np.arange(len(matrix.columns)), xticklabels=matrix.columns,
           yticks=np.arange(len(matrix)), yticklabels=matrix.index, title=title)
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
    if annotate:
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f'{value:+.2f}' if np.isfinite(value) else 'N/A',
                    ha='center', va='center', fontsize=8,
                    color='white' if np.isfinite(value) and abs(value) > .55*vmax else 'black')
    fig.colorbar(artist, ax=ax, label='Δ score = Chandra − raw', shrink=.9)
    return fig


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class Run:
    path: Path
    metadata: dict
    metrics: dict
    splits: dict
    status: str

    @property
    def id(self):
        return str(self.path)  # Identité unique même avec plusieurs racines.

    @property
    def config(self):
        return self.metadata.get('config', self.metrics.get('config', {}))

    @property
    def classes(self):
        return self.metrics.get('class_names', self.metadata.get('class_names', []))

    def predictions(self, split='test', cohort='all'):
        path = self.path / f'{split}_predictions.csv'
        if not path.exists():
            raise ValueError(f'{self.path.name}: {path.name} missing')
        frame = pd.read_csv(path, dtype={'key': str, 'cluster_id': str})
        required = ['key', 'cluster_id', 'label', 'prediction', 'has_chandra']
        required += [f'probability_{c}' for c in self.classes]
        if not self.classes or not set(required) <= set(frame):
            raise ValueError('Missing prediction columns or class order')
        if frame.key.isna().any() or frame.key.duplicated().any() or frame.cluster_id.isna().any():
            raise ValueError('Missing identifiers or duplicate keys')
        flag = frame.has_chandra.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
        if flag.isna().any():
            raise ValueError('Invalid has_chandra value')
        frame['has_chandra'] = flag.astype(bool)
        probabilities = frame[[f'probability_{c}' for c in self.classes]].to_numpy(float)
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or not np.allclose(probabilities.sum(1), 1, atol=1e-5)):
            raise ValueError('Invalid probabilities')
        for column in ['label', 'prediction']:
            if not frame[column].isin(range(len(self.classes))).all():
                raise ValueError(f'{column} outside the class index range')
        if not np.array_equal(frame.prediction, probabilities.argmax(1)):
            raise ValueError('prediction differs from argmax(probabilities)')
        if cohort == 'paired':
            frame = frame[frame.has_chandra]
        elif cohort == 'raw_only':
            frame = frame[~frame.has_chandra]
        elif cohort != 'all':
            raise ValueError('Unknown cohort')
        return frame.set_index('key', drop=False).sort_index()

    def history(self):
        path = self.path / 'history.csv'
        if path.exists():
            return pd.read_csv(path)
        path = self.path / 'history.jsonl'
        rows = []
        if path.exists():
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                row = {k: record.get(k) for k in ['epoch', 'seconds', 'learning_rate']}
                for source, prefix in [('train', 'train'), ('validation', 'val')]:
                    row.update({f'{prefix}_{k}': v for k, v in record.get(source, {}).items()
                                if v is None or np.isscalar(v)})
                rows.append(row)
        return pd.DataFrame(rows)


def discover(roots):
    """Inclut aussi les exécutions échouées/incomplètes ; rapporte les JSON invalides."""
    paths, runs, issues = set(), {}, []
    for root in map(Path, roots):
        if not root.exists():
            issues.append({'path': str(root), 'error': 'Root directory missing'})
            continue
        for name in ['config.json', 'metrics.json', 'status.json']:
            paths.update(p.parent.resolve() for p in root.rglob(name))
    for path in sorted(paths):
        try:
            meta, metrics = read_json(path/'config.json'), read_json(path/'metrics.json')
            if not (meta.get('config', metrics.get('config', {})).get('model')):
                continue
            status = read_json(path/'status.json').get('status', 'completed' if metrics else 'incomplete')
            run = Run(path, meta, metrics, read_json(path/'splits.json'), status)
            runs[run.id] = run
        except (ValueError, OSError, TypeError, AttributeError) as exc:
            issues.append({'path': str(path), 'error': str(exc)})
    rows = []
    for run in runs.values():
        row = dict(run_id=run.id, name=run.path.name, status=run.status,
                   model=run.config.get('model'), mode=run.config.get('mode'),
                   seed=run.config.get('seed'), date=run.metadata.get('created_at'),
                   parameters=run.metadata.get('parameters'),
                   best_epoch=run.metrics.get('best_epoch'), epochs=run.metrics.get('epochs_ran'),
                   minutes=(run.metrics.get('seconds') or 0)/60)
        row.update({f'{s}_{k}': run.metrics.get(s, {}).get(k) for s in ['validation', 'test'] for k in METRICS[:6]})
        rows.append(row)
    return runs, pd.DataFrame(rows), pd.DataFrame(issues)


def scores(frame, classes):
    if frame.empty:
        return {k: np.nan for k in METRICS} | {'n': 0, 'n_clusters': 0}
    y = frame.label.to_numpy(int)
    p = frame[[f'probability_{c}' for c in classes]].to_numpy(float)
    pred = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=range(len(classes)), zero_division=0)
    auc, ap = [], []
    for i in range(len(classes)):
        binary = y == i
        present = 0 < binary.sum() < len(y)
        auc.append(roc_auc_score(binary, p[:, i]) if present else np.nan)
        ap.append(average_precision_score(binary, p[:, i]) if present else np.nan)
    return dict(n=len(y), n_clusters=frame.cluster_id.nunique(), accuracy=accuracy_score(y, pred),
                balanced_accuracy=recall[support > 0].mean(), macro_f1=f1.mean(),
                weighted_f1=np.average(f1, weights=support), roc_auc_macro=np.mean(auc),
                average_precision_macro=np.mean(ap),
                log_loss=-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean(),
                brier=np.square(p - np.eye(len(classes))[y]).sum(axis=1).mean())


def cohort_id(run, frame):
    sources = run.splits.get('source_signatures', {})
    if not sources or any(k not in sources for k in frame.index):
        return None
    return fingerprint({'classes': run.classes, 'samples': [
        (k, row.cluster_id, int(row.label), sources[k]) for k, row in frame.iterrows()]})


def comparison(runs, split='validation', cohort='all', common=False):
    frames, issues = {}, []
    for run in runs:
        if run.status != 'completed':
            issues.append({'run': run.path.name, 'error': f'Status {run.status}: excluded'})
            continue
        try:
            frame = run.predictions(split, cohort)
            if frame.empty:
                raise ValueError('Empty cohort')
            frames[run.id] = frame
        except (ValueError, OSError) as exc:
            issues.append({'run': run.path.name, 'error': str(exc)})
    if common and frames:
        keys = sorted(set.intersection(*(set(f.index) for f in frames.values())))
        if not keys:
            return pd.DataFrame(), pd.DataFrame(issues + [{'error': 'Empty intersection'}]), {}
        frames = {k: f.loc[keys] for k, f in frames.items()}
    rows = []
    for run in runs:
        if run.id not in frames:
            continue
        frame = frames[run.id]
        cid = cohort_id(run, frame)
        # Provenance inconnue : groupe propre au run, jamais une égalité implicite.
        group = cid or f'unverified:{run.id}'
        rows.append(dict(run_id=run.id, name=run.path.name, model=run.config.get('model'),
                         mode=run.config.get('mode'), seed=run.config.get('seed'),
                         cohort_id=group, provenance_verified=cid is not None,
                         **scores(frame, run.classes)))
    return pd.DataFrame(rows), pd.DataFrame(issues), frames


def rank_table(table, metric='macro_f1'):
    if table.empty:
        return table.copy()
    result = table.copy()
    result['rank_in_cohort'] = result.groupby('cohort_id')[metric].rank(
        ascending=metric in LOWER, method='min', na_option='keep')
    return result.sort_values(['cohort_id', 'rank_in_cohort', 'name'])


def plot_comparison(table, metric='macro_f1'):
    if table.empty:
        return
    for cohort, group in table.groupby('cohort_id', sort=False):
        group = group.sort_values(metric, ascending=metric not in LOWER)
        fig, ax = plt.subplots(figsize=(11, max(3, .3*len(group))))
        ax.barh(group.name, group[metric])
        ax.set(xlabel=metric, title=f'Cohort {cohort[:16]} — n={group.n.iloc[0]}')
        ax.grid(axis='x', alpha=.2)
        fig.tight_layout()
        plt.show()


def detail(run, split='test', display_fn=print):
    """Configuration, évolution, classes, calibration, cohortes et erreurs."""
    display_fn(pd.Series(run.config, name='configuration').to_frame())
    display_fn(pd.Series({k: run.metadata.get(k) for k in ['model_class', 'parameters', 'versions', 'normalization']}).to_frame('metadata'))
    history = run.history()
    if not history.empty:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, metric in zip(axes.flat, ['loss', 'macro_f1', 'accuracy', 'roc_auc_macro', 'balanced_accuracy', 'learning_rate']):
            cols = [metric] if metric == 'learning_rate' else [f'train_{metric}', f'val_{metric}']
            for col in cols:
                if col in history:
                    ax.plot(history.epoch, history[col], label=col)
            if run.metrics.get('best_epoch'):
                ax.axvline(run.metrics['best_epoch'], color='black', ls=':', label='checkpoint')
            ax.set(title=metric, xlabel='Epoch')
            if ax.lines:
                ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.suptitle(run.path.name)
        fig.tight_layout()
        plt.show()
    rows = []
    for s, c in [('validation', 'all'), ('test', 'all'), ('test', 'paired'), ('test', 'raw_only')]:
        try:
            rows.append({'split': s, 'cohort': c, **scores(run.predictions(s, c), run.classes)})
        except (ValueError, OSError) as exc:
            print(exc)
    display_fn(pd.DataFrame(rows))
    try:
        f = run.predictions(split)
    except (ValueError, OSError) as exc:
        print(exc)
        return
    if f.empty:
        print('No predictions available.')
        return
    y = f.label.to_numpy(int)
    p = f[[f'probability_{c}' for c in run.classes]].to_numpy(float)
    pred = p.argmax(1)
    pr, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=range(len(run.classes)), zero_division=0)
    display_fn(pd.DataFrame({'precision': pr, 'recall': rec, 'f1': f1, 'support': sup}, index=run.classes))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    cm = confusion_matrix(y, pred, labels=range(len(run.classes)))
    norm = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(1, keepdims=True) != 0)
    for ax, values, title in zip(axes[0, :2], [cm, norm], ['Confusion matrix — counts', 'Confusion matrix — per-class recall']):
        ax.imshow(values, cmap='Blues', vmin=0)
        for (i, j), v in np.ndenumerate(values):
            ax.text(j, i, f'{v:.2f}' if values is norm else str(v), ha='center')
        ax.set(xticks=range(len(run.classes)), yticks=range(len(run.classes)),
               xticklabels=run.classes, yticklabels=run.classes, xlabel='Predicted', ylabel='True', title=title)
    for i, cls in enumerate(run.classes):
        binary = y == i
        if 0 < binary.sum() < len(y):
            x, z, _ = roc_curve(binary, p[:, i])
            axes[0, 2].plot(x, z, label=f'{cls}: {roc_auc_score(binary,p[:,i]):.3f}')
            precision, recall, _ = precision_recall_curve(binary, p[:, i])
            axes[1, 0].plot(recall, precision, label=f'{cls}: AP={average_precision_score(binary,p[:,i]):.3f}')
            true, predicted = calibration_curve(binary, p[:, i], n_bins=6, strategy='quantile')
            axes[1, 1].plot(predicted, true, 'o-', label=cls)
    axes[0, 2].plot([0, 1], [0, 1], 'k:', label='Chance')
    axes[0, 2].set(title='ROC one-vs-rest', xlabel='FPR', ylabel='TPR')
    axes[1, 0].set(title='Precision–recall', xlabel='Recall', ylabel='Precision')
    axes[1, 1].plot([0, 1], [0, 1], 'k:', label='Ideal')
    axes[1, 1].set(title='Calibration', xlabel='Predicted probability', ylabel='Observed frequency')
    for good, name in [(True, 'Correct'), (False, 'Error')]:
        axes[1, 2].hist(p.max(1)[(y == pred) == good], bins=np.linspace(0, 1, 11), alpha=.6, label=name)
    axes[1, 2].set(title='Confidence', xlabel='Maximum probability', ylabel='Count')
    for ax in [axes[0, 2], *axes[1]]:
        if ax.lines or ax.patches:
            ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    errors = f.assign(confidence=p.max(1), correct=y == pred)
    display_fn(errors.loc[~errors.correct].sort_values('confidence', ascending=False))
    print('Chandra availability × true class (availability ≠ model use)')
    display_fn(pd.crosstab(f.has_chandra, f.label.map(dict(enumerate(run.classes)))))


# Différences nécessaires/techniques, exclues du contrôle des hyperparamètres.
IGNORED_CONFIG = {'name', 'mode', 'cache', 'processed_dir', 'verbose', 'num_workers', 'device'}


def pair_audit(raw, chandra):
    reasons = []
    if raw.config.get('mode') != 'raw' or chandra.config.get('mode') not in {'pair', 'multimodal'}:
        reasons.append('incompatible modes')
    for key in sorted((set(raw.config) | set(chandra.config)) - IGNORED_CONFIG):
        if raw.config.get(key) != chandra.config.get(key):
            reasons.append(f'config.{key}')
    if raw.classes != chandra.classes:
        reasons.append('class order')
    for key in ['model_class', 'library_source_sha256', 'versions', 'normalization']:
        if not raw.metadata.get(key) or raw.metadata.get(key) != chandra.metadata.get(key):
            reasons.append(f'metadata.{key}')
    for split in ['train', 'validation', 'test']:
        a, b = raw.splits.get(split), chandra.splits.get(split)
        if a is None or b is None or sorted(a) != sorted(b):
            reasons.append(f'split.{split}')
    a, b = raw.splits.get('source_signatures'), chandra.splits.get('source_signatures')
    if not a or not b or a != b:
        reasons.append('missing/different sources')
    return reasons


def chandra_pairs(runs):
    """Toutes les références du même modèle ; aucune sélection opportuniste du meilleur raw."""
    rows = []
    for chandra in runs:
        if chandra.config.get('mode') not in {'pair', 'multimodal'} or chandra.status != 'completed':
            continue
        baselines = [r for r in runs if r.config.get('mode') == 'raw'
                     and r.config.get('model') == chandra.config.get('model') and r.status == 'completed']
        if not baselines:
            rows.append(dict(chandra_id=chandra.id, raw_id=None, model=chandra.config.get('model'),
                             controlled=False, differences='No raw baseline selected'))
        for raw in baselines:
            reasons = pair_audit(raw, chandra)
            rows.append(dict(chandra_id=chandra.id, raw_id=raw.id, model=chandra.config.get('model'),
                             controlled=not reasons, differences=', '.join(reasons)))
    return pd.DataFrame(rows)


def paired_frames(raw, chandra, split, cohort):
    if raw.classes != chandra.classes:
        raise ValueError('Different class orders')
    a, b = raw.predictions(split, cohort), chandra.predictions(split, cohort)
    keys = sorted(set(a.index) & set(b.index))
    if not keys:
        raise ValueError('No shared samples')
    a, b = a.loc[keys], b.loc[keys]
    for col in ['label', 'cluster_id', 'has_chandra']:
        if not np.array_equal(a[col], b[col]):
            raise ValueError(f'Mismatch in {col}')
    # Ne pas évaluer un amas appris par l'autre entraînement.
    for run in [raw, chandra]:
        forbidden = set(run.splits.get('train', []))
        if split == 'test':
            forbidden.update(run.splits.get('validation', []))
        samples = run.splits.get('samples', {})
        forbidden_clusters = {samples[k].get('cluster_id') for k in forbidden if k in samples}
        if forbidden & set(keys) or forbidden_clusters & set(a.cluster_id):
            raise ValueError('Split leakage for the compared samples')
    return a, b


def paired_delta(raw, chandra, split='test', cohort='paired', n_boot=500, seed=42):
    """Bootstrap apparié par amas ; incertitude d'échantillonnage, pas d'entraînement."""
    a, b = paired_frames(raw, chandra, split, cohort)
    sa, sb = scores(a, raw.classes), scores(b, chandra.classes)
    groups = [np.flatnonzero(a.cluster_id.to_numpy() == c) for c in a.cluster_id.unique()]
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in METRICS}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        ma, mb = scores(a.iloc[idx], raw.classes), scores(b.iloc[idx], chandra.classes)
        for k in METRICS:
            draws[k].append(mb[k] - ma[k])
    rows = []
    for k in METRICS:
        valid = np.asarray(draws[k], dtype=float)
        valid = valid[np.isfinite(valid)]
        ci = np.quantile(valid, [.025, .975]) if len(valid) >= 2 else [np.nan, np.nan]
        delta = sb[k] - sa[k]
        rows.append(dict(metric=k, raw=sa[k], chandra=sb[k], delta=delta,
                         gain_oriented=-delta if k in LOWER else delta,
                         ci_low=ci[0], ci_high=ci[1], valid_bootstrap=len(valid),
                         n=len(a), n_clusters=len(groups), cohort=cohort))
    return pd.DataFrame(rows)


def protocol_summary(table, runs, metric='macro_f1'):
    """Moyenne par cohorte puis entre cohortes ; répétitions non surpondérées."""
    if table.empty:
        return pd.DataFrame()
    lookup = {r.id: r for r in runs}
    annotated = table.copy()
    def protocol(run_id):
        run = lookup[run_id]
        config = {k: v for k, v in run.config.items()
                  if k not in (IGNORED_CONFIG - {'mode'}) | {'seed'}}
        return fingerprint({'config': config, 'classes': run.classes,
                            'metadata': {k: run.metadata.get(k) for k in
                                ['model_class', 'library_source_sha256', 'normalization', 'versions']}})
    annotated['protocol_id'] = annotated.run_id.map(protocol)
    rows = []
    for pid, group in annotated.groupby('protocol_id'):
        per_cohort = group.groupby('cohort_id')[METRICS].mean()
        row = dict(protocol_id=pid, model=group.model.iloc[0], mode=group['mode'].iloc[0],
                   n_runs=len(group), n_cohorts=len(per_cohort),
                   cohort_set=fingerprint(sorted(group.cohort_id.unique())),
                   seeds=sorted({str(lookup[r].config.get('seed')) for r in group.run_id}))
        for name in METRICS:
            row[f'{name}_mean'] = per_cohort[name].mean()
            row[f'{name}_std'] = per_cohort[name].std(ddof=1)
            row[f'{name}_valid_cohorts'] = int(per_cohort[name].notna().sum())
        rows.append(row)
    result = pd.DataFrame(rows)
    # Une métrique absente sur une cohorte ne suffit pas pour classer la configuration.
    eligible = result[f'{metric}_valid_cohorts'] == result.n_cohorts
    result['rank_in_cohort_set'] = np.nan
    result.loc[eligible, 'rank_in_cohort_set'] = result[eligible].groupby('cohort_set')[f'{metric}_mean'].rank(
        ascending=metric in LOWER, method='min')
    return result.sort_values(['cohort_set', 'rank_in_cohort_set'])
