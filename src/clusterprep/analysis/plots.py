"""Bounded, opt-in figures; FITS/gallery work delegates to the existing renderer."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from .registry import epoch_table, prediction_table
from .comparison import aggregate_history, aggregate_metrics


def plot_history(runs, metric='loss', aggregate=False):
    epochs = epoch_table(runs)
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    if epochs.empty:
        ax.set_title('No history available')
        return fig
    for prefix, style in [('train', '--'), ('validation', '-')]:
        column = f'{prefix}.{metric}'
        if column not in epochs:
            continue
        if aggregate:
            table = aggregate_history(epochs, column)
            for (config, protocol, phase), group in table.groupby(['configuration_id', 'protocol_id', 'phase']):
                x = group.phase_epoch.to_numpy(float)
                mean = group['mean'].to_numpy(float)
                std = group['std'].to_numpy(float)
                run = next(r for r in runs if r.configuration_id == config)
                ax.plot(x, mean, style, label=f'{run.configuration_name} / {phase} / {prefix}')
                ax.fill_between(x, mean-std, mean+std, alpha=.15)
        else:
            for run in runs:
                group = epochs[epochs.run_id == run.run_id]
                label = run.run_name
                ax.plot(group.epoch, group[column], style, label=f'{label} / {prefix}')
                if prefix == 'validation' and run.metrics.get('best_epoch') is not None:
                    ax.axvline(run.metrics['best_epoch'], color='gray', alpha=.3, linestyle=':')
                if prefix == 'validation' and run.metrics.get('stopped_early') and not group.empty:
                    ax.scatter(group.epoch.iloc[-1], group[column].iloc[-1], marker='x', color='red')
                for _, phase in group.groupby('phase', sort=False):
                    if phase.epoch.min() != group.epoch.min():
                        ax.axvline(phase.epoch.min(), color='orange', alpha=.4)
    ax.set(xlabel='Epoch within phase (available folds only)' if aggregate else 'Global epoch', ylabel=metric)
    if ax.lines:
        ax.legend(fontsize=6)
    return fig


def plot_performance(table, metric='macro_f1', kind='bar'):
    """Facets by compatibility protocol; never put binary and multilabel on one axis."""
    if table.empty or metric not in table:
        fig, ax = plt.subplots(); ax.set_title(f'{metric}: unavailable'); return fig
    table = table[(table.status == 'completed') & table[metric].notna()]
    groups = list(table.groupby(['protocol_id', 'metric_source'], sort=True))
    fig, axes = plt.subplots(max(1, len(groups)), 1, figsize=(10, max(4, 3.5 * len(groups))), squeeze=False, layout='constrained')
    for ax, ((protocol, source), group) in zip(axes.flat, groups):
        configs = list(group.configuration_id.unique())
        labels = ['\n'.join(group.loc[group.configuration_id == c, 'configuration_name'].unique()) for c in configs]
        if kind == 'heatmap':
            data = group.pivot_table(index='configuration_id', columns='fold', values=metric).reindex(configs)
            if data.shape[1]:
                heatmap = ax.imshow(data, aspect='auto')
                ax.set_yticks(range(len(configs)), labels)
                for i, row in enumerate(data.to_numpy()):
                    for j, value in enumerate(row):
                        if pd.notna(value):
                            ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                                    color='black' if heatmap.norm(value) > .5 else 'white')
                ax.set_xticks(range(len(data.columns)), data.columns); ax.set_xlabel('Fold')
        elif kind == 'strip':
            for i, config in enumerate(configs):
                values = group[group.configuration_id == config][metric]
                ax.scatter(np.full(len(values), i), values)
            ax.set_xticks(range(len(configs)), labels)
        else:
            summary = aggregate_metrics(group, [metric]).set_index('configuration_id').reindex(configs)
            ax.bar(range(len(configs)), summary[metric+'_mean'], yerr=summary[metric+'_std'].fillna(0))
            ax.set_xticks(range(len(configs)), [f'{label}\nn_folds={int(summary.loc[c, "n_folds"])}' for c, label in zip(configs, labels)])
        ax.set_title(f'{group.iloc[0].task or group.iloc[0].task_type} / {group.iloc[0].split} / protocol {protocol[:8]} — {metric}')
    return fig


def _plot_diagnostic_history(run, axes):
    """One shared history row, including for runs without detailed predictions."""
    history = epoch_table([run])
    for ax, metric, title in zip(axes, ('loss', 'macro_f1', 'accuracy'), ('Loss', 'Macro-F1', 'Accuracy')):
        available = False
        for split, label, style in [('train', 'Entraînement', '--'), ('validation', 'Validation', '-')]:
            column = f'{split}.{metric}'
            if column not in history or not history[column].notna().any():
                continue
            ax.plot(history.epoch, history[column], style, label=label)
            available = True
        if available:
            best = run.metrics.get('best_epoch')
            if best is not None:
                ax.axvline(best, color='gray', linestyle=':', alpha=.7, label='Meilleure epoch')
            for _, phase in list(history.groupby('phase', sort=False))[1:]:
                ax.axvline(phase.epoch.iloc[0], color='orange', alpha=.5, linestyle=':')
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, 'Historique indisponible', ha='center', transform=ax.transAxes)
        ax.set(title=f'{title} par epoch', xlabel='Epoch', ylabel=title)
        ax.grid(alpha=.2)


def plot_diagnostics(run, split='test', cohort='all'):
    table = prediction_table([run], split, cohort)
    labels = run.diagnostic_labels
    diagnostic_rows = max(1, len(labels))
    fig = plt.figure(figsize=(16, 3.5 * (diagnostic_rows + 1)), layout='constrained')
    grid = fig.add_gridspec(diagnostic_rows + 1, 12)
    axes = [[fig.add_subplot(grid[row, col * 3:(col + 1) * 3]) for col in range(4)]
            for row in range(diagnostic_rows)]
    history_axes = [fig.add_subplot(grid[-1, col * 4:(col + 1) * 4]) for col in range(3)]
    _plot_diagnostic_history(run, history_axes)
    for axes_row, label in zip(axes, labels):
        if table.empty:
            for ax in axes_row:
                ax.text(.5, .5, 'Prédictions détaillées indisponibles', ha='center', transform=ax.transAxes)
                ax.set_axis_off()
            continue
        frame = table[(table.label == label) & table.known]
        y, score = frame.truth.to_numpy(), frame.score.to_numpy()
        if len(frame) and frame.prediction.notna().all():
            matrix = confusion_matrix(y, frame.prediction, labels=[0, 1])
            support = matrix.sum(axis=1, keepdims=True)
            percentages = np.divide(100.0 * matrix, support,
                                    out=np.zeros_like(matrix, dtype=float), where=support != 0)
            axes_row[0].imshow(percentages, cmap='Blues', vmin=0, vmax=100)
            for i in range(2):
                for j in range(2):
                    text = f'{percentages[i, j]:.1f}% ({matrix[i, j]})' if support[i, 0] else '— (0)'
                    axes_row[0].text(j, i, text, ha='center', va='center',
                                     color='white' if percentages[i, j] > 50 else 'black')
        else:
            axes_row[0].text(.5, .5, 'Thresholds or known labels unavailable', ha='center', wrap=True)
        negative = next((name for name in run.labels if name != label), f'non_{label}') if run.task_type == 'binary' else f'non_{label}'
        class_names = [negative, label]
        axes_row[0].set(title=f'{label}: confusion (% par classe réelle)',
                        xlabel='Prédiction', ylabel='Classe réelle',
                        xticks=[0, 1], yticks=[0, 1],
                        xticklabels=class_names, yticklabels=class_names)
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, score)
            precision, recall, _ = precision_recall_curve(y, score)
            axes_row[1].plot(fpr, tpr)
            axes_row[2].plot(recall, precision)
            axes_row[2].axhline(np.mean(y), linestyle=':', color='gray')
        axes_row[1].set(title='ROC', xlabel='FPR', ylabel='TPR')
        axes_row[2].set(title='Precision–Recall', xlabel='Recall', ylabel='Precision')
        for truth in (0, 1):
            axes_row[3].hist(score[y == truth], bins=np.linspace(0, 1, 11), alpha=.5, label=f'réel : {class_names[truth]}')
        axes_row[3].set(title='Probabilities', xlabel='Score')
        axes_row[3].legend()
    fig.suptitle(f'{run.run_name} / {split}')
    return fig


def plot_galleries(run, *, processed_root, split='test', cohort='all', max_clusters=12,
                   radio_template='{cluster_id}/{cluster_id}_RAW_processed_size128_v2.fits',
                   chandra_template='{cluster_id}/{cluster_id}_CHANDRA_processed_size128_v2.fits',
                   chandra_only=False, view="overlay"):
    from ..multilabel_analysis import plot_classification_gallery
    if view not in ('overlay', 'radio', 'chandra'):
        raise ValueError('Gallery view: overlay, radio or chandra')
    frame = prediction_table([run], split, cohort)
    figures = {}
    if frame.empty:
        return figures
    if run.task_type == 'multilabel':
        groups = [(f'{label}_{outcome}', frame[(frame.label == label) & (frame.correct == correct)])
                  for label in run.labels for outcome, correct in [('correct', True), ('incorrect', False)]]
    else:
        label = run.diagnostic_labels[0]
        data = frame[frame.label == label]
        groups = [(name, data[(data.truth == y) & (data.prediction == p)])
                  for name, y, p in [('TP', 1, 1), ('TN', 0, 0), ('FP', 0, 1), ('FN', 1, 0)]]
    for name, data in groups:
        entries = []
        for row in data.head(max_clusters).to_dict('records'):
            # Existing renderer understands either binary or RR/RH title fields.
            row.update(positive_class=row['label'], probability=row['score'], label=int(row['truth']))
            if view == 'radio':
                row['has_chandra'] = False
            entries.append(row)
        figures[name] = plot_classification_gallery(entries, title=f'{run.run_name} / {name}',
            processed_root=processed_root, radio_template=radio_template,
            chandra_template=chandra_template, chandra_only=chandra_only or view == 'chandra')
    return figures


def plot_paired(table):
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    if not table.empty:
        paired = table[table.paired]
        ax.scatter(range(len(paired)), paired.delta)
        ax.set_xticks(range(len(paired)), [f'{r.left_name} → {r.right_name}' for r in paired.itertuples()], rotation=45)
    ax.axhline(0, color='gray'); ax.set(ylabel='Paired metric difference (right − left)')
    return fig
