"""Unified, artifact-only binary and multilabel training analysis."""
from .registry import (TrainingRun, discover_runs, load_run, filter_runs, registry_table,
                       epoch_table, prediction_table)
from .comparison import (metrics_table, metrics_by_fold, aggregate_metrics, aggregate_history, assert_comparable,
                         paired_comparisons, compare_predictions, transfer_table, out_of_fold_predictions)
from .plots import (plot_history, plot_performance, plot_diagnostics, plot_galleries, plot_paired)
from .export import export_analysis

from .navigation import filter_controls, display_table
