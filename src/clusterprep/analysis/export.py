"""Collision-free exports outside source training directories."""
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

from .registry import registry_table, epoch_table, prediction_table
from .comparison import metrics_table, aggregate_metrics, transfer_table, metrics_by_fold


def export_analysis(runs, root, *, split='test', cohort='all', metrics=('macro_f1', 'roc_auc_macro'),
                    figures=None, export_tables=True, settings=None):
    root = Path(root).expanduser().resolve()
    if any(root == r.path or root.is_relative_to(r.path) for r in runs):
        raise ValueError('Export directory must be outside training runs')
    destination = root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    table = metrics_table(runs, split, cohort)
    if export_tables:
        tables = {'run_registry': registry_table(runs), 'metrics_by_run': table,
                  'metrics_by_model': aggregate_metrics(table, metrics),
                  'metrics_by_fold': metrics_by_fold(table, metrics), 'epochs': epoch_table(runs),
                  'predictions': prediction_table(runs, split, cohort), 'transfer': transfer_table(runs)}
        for name, frame in tables.items():
            frame.to_csv(destination / f'{name}.csv', index=False)
    for name, figure in (figures or {}).items():
        target = (destination / 'figures' / f'{name}.png').resolve()
        if not target.is_relative_to(destination):
            raise ValueError('Invalid figure export name')
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=130, bbox_inches='tight')
    report = destination/'reports'
    report.mkdir()
    (report/'selection.json').write_text(json.dumps(dict(split=split, cohort=cohort, metrics=list(metrics),
        runs=[r.summary() for r in runs], settings=settings or {},
        policy='Completed runs only in aggregates; sample std; available epochs; no OOF pooling.'), indent=2, default=str))
    return destination
