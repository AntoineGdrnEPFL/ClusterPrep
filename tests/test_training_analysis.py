"""Vérifie les garde-fous statistiques du notebook, sans entraînement."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
from training_analysis import (Run, scores, discover, comparison, rank_table,
                               chandra_pairs, pair_audit, paired_delta, protocol_summary,
                               gain_scores, gain_tables, plot_gain_heatmap)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_fixture(self, name='raw', mode='raw'):
        path = self.root / name
        path.mkdir()
        frame = pd.DataFrame(dict(key=['a', 'b', 'c', 'd'], cluster_id=['A', 'B', 'C', 'D'],
            label=[0, 0, 1, 1], prediction=[0, 1, 1, 1], has_chandra=[True, True, False, False],
            probability_DE=[.9, .4, .2, .1], probability_NDE=[.1, .6, .8, .9]))
        frame.to_csv(path/'test_predictions.csv', index=False)
        frame.to_csv(path/'validation_predictions.csv', index=False)
        meta = dict(config=dict(model='cnn', mode=mode, seed=42), class_names=['DE', 'NDE'],
                    model_class='CNN', library_source_sha256='abc', versions={'numpy': '1'},
                    normalization={'alpha': 10})
        splits = dict(train=['train'], validation=['val'], test=['a', 'b', 'c', 'd'],
                      source_signatures={k: {'raw_path': k} for k in ['train', 'val', 'a', 'b', 'c', 'd']})
        return Run(path, meta, {}, splits, 'completed')

    def test_scores_match_known_confusion(self):
        run = self.run_fixture()
        result = scores(run.predictions(), run.classes)
        self.assertEqual(result['accuracy'], .75)
        self.assertAlmostEqual(result['macro_f1'], (2/3 + 4/5)/2)
        self.assertAlmostEqual(result['balanced_accuracy'], .75)

    def test_gain_matrix_sign_and_de_class(self):
        a, b = self.run_fixture(), self.run_fixture('chandra', 'multimodal')
        f = pd.read_csv(b.path/'test_predictions.csv')
        f.loc[1, ['prediction', 'probability_DE', 'probability_NDE']] = [0, .8, .2]
        f.to_csv(b.path/'test_predictions.csv', index=False)
        matrix, details, issues = gain_tables([a, b], cohort='all')
        self.assertEqual(len(matrix), 1)
        self.assertAlmostEqual(matrix.iloc[0].accuracy, .25)
        self.assertAlmostEqual(matrix.iloc[0].recall_de, .5)
        self.assertAlmostEqual(matrix.iloc[0].f1_de, 1/3)
        self.assertTrue(details.controlled.all())
        self.assertTrue(issues.empty)
        frame = a.predictions()
        frame['label'] = 1 - frame.label
        self.assertEqual(gain_scores(frame, ['NDE', 'DE'])['recall_de'], .5)
        fig = plot_gain_heatmap(matrix)
        self.assertGreater(fig.axes[0].images[0].cmap(1.0)[2], fig.axes[0].images[0].cmap(1.0)[0])
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_pair_raw_filter_allowed_but_other_protocol_changes_rejected(self):
        a, b = self.run_fixture(), self.run_fixture('pair', 'pair')
        for run in [a, b]:
            run.splits['base_splits'] = {s: list(run.splits[s]) for s in ['train', 'validation', 'test']}
            run.splits['samples'] = {k: {'chandra_path': k if k in ['a', 'b', 'val'] else None}
                                     for k in run.splits['source_signatures']}
        b.splits.update(train=[], validation=['val'], test=['a', 'b'])
        f = pd.read_csv(b.path/'test_predictions.csv').iloc[:2]
        f.to_csv(b.path/'test_predictions.csv', index=False)
        matrix, details, _ = gain_tables([a, b])
        self.assertEqual(len(matrix), 1)
        self.assertFalse(details.controlled.iloc[0])
        self.assertEqual(details.n.iloc[0], 2)
        self.assertTrue(np.isnan(matrix.iloc[0].roc_auc))
        b.metadata['library_source_sha256'] = 'new-code'
        matrix, details, _ = gain_tables([a, b], allow_code_change_models=('cnn',))
        self.assertEqual(len(matrix), 1)
        self.assertTrue(details.code_changed.iloc[0])
        self.assertTrue(details.training_population_changed.iloc[0])
        b.metadata['library_source_sha256'] = 'abc'
        b.config['seed'] = 7
        self.assertTrue(gain_tables([a, b])[0].empty)
        self.assertTrue(gain_tables([b])[0].empty)

    def test_single_class_auc_undefined(self):
        run = self.run_fixture()
        self.assertTrue(np.isnan(scores(run.predictions(cohort='paired'), run.classes)['roc_auc_macro']))

    def test_code_change_requires_explicit_model_opt_in(self):
        a, b = self.run_fixture(), self.run_fixture('chandra', 'multimodal')
        a.config['model'] = b.config['model'] = 'dual_encoder_cnn'
        b.metadata['library_source_sha256'] = 'new-code'
        self.assertTrue(gain_tables([a, b])[0].empty)
        kwargs = dict(allow_code_change_models=('dual_encoder_cnn',))
        matrix, details, _ = gain_tables([a, b], **kwargs)
        self.assertEqual(len(matrix), 1)
        self.assertTrue(details.code_changed.iloc[0])
        self.assertFalse(details.controlled.iloc[0])
        self.assertIn('[code change]', matrix.index[0])
        self.assertEqual(details.raw_code_sha256.iloc[0], 'abc')
        self.assertTrue(gain_tables([a, b], allow_code_change_models=('cnn',))[0].empty)
        for field, value in [('seed', 7), ('size', 64)]:
            b.config[field] = value
            self.assertTrue(gain_tables([a, b], **kwargs)[0].empty)
            if field in a.config:
                b.config[field] = a.config[field]
            else:
                del b.config[field]
        b.splits['source_signatures']['a'] = {'raw_path': 'different'}
        self.assertTrue(gain_tables([a, b], **kwargs)[0].empty)
        b.splits['source_signatures']['a'] = a.splits['source_signatures']['a']
        b.metadata.pop('library_source_sha256')
        self.assertTrue(gain_tables([a, b], **kwargs)[0].empty)

    def test_duplicate_keys_rejected(self):
        run = self.run_fixture()
        frame = pd.read_csv(run.path/'test_predictions.csv')
        frame.loc[1, 'key'] = 'a'
        frame.to_csv(run.path/'test_predictions.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            run.predictions()

    def test_identical_models_zero_paired_interval(self):
        a, b = self.run_fixture(), self.run_fixture('chandra', 'multimodal')
        self.assertEqual(pair_audit(a, b), [])
        delta = paired_delta(a, b, cohort='all', n_boot=10)
        self.assertTrue((delta.delta == 0).all())
        self.assertTrue((delta.ci_low == 0).all())
        self.assertTrue((delta.ci_high == 0).all())

    def test_protocol_mismatch_and_missing_baseline(self):
        a, b = self.run_fixture(), self.run_fixture('chandra', 'multimodal')
        b.config['seed'] = 7
        self.assertIn('config.seed', pair_audit(a, b))
        self.assertFalse(chandra_pairs([a, b]).controlled.any())
        self.assertIsNone(chandra_pairs([b]).iloc[0].raw_id)

    def test_provenance_separates_rankings(self):
        a, b = self.run_fixture(), self.run_fixture('chandra', 'multimodal')
        b.splits['source_signatures']['a'] = {'raw_path': 'changed'}
        table, _, _ = comparison([a, b])
        self.assertEqual(table.cohort_id.nunique(), 2)
        self.assertTrue((rank_table(table).rank_in_cohort == 1).all())

    def test_missing_provenance_not_assumed_equal(self):
        a, b = self.run_fixture(), self.run_fixture('chandra', 'multimodal')
        a.splits = b.splits = {}
        table, _, _ = comparison([a, b])
        self.assertEqual(table.cohort_id.nunique(), 2)
        self.assertFalse(table.provenance_verified.any())

    def test_protocol_summary_does_not_overweight_repeated_cohorts(self):
        a = self.run_fixture()
        table, _, _ = comparison([a])
        repeated = pd.concat([table, table, table], ignore_index=True)
        repeated['cohort_id'] = ['first', 'first', 'second']
        repeated['macro_f1'] = [1.0, 1.0, 0.0]
        result = protocol_summary(repeated, [a])
        self.assertEqual(result.iloc[0].macro_f1_mean, .5)
        self.assertEqual(result.iloc[0].n_cohorts, 2)
        repeated.loc[2, 'macro_f1'] = np.nan
        result = protocol_summary(repeated, [a])
        self.assertTrue(np.isnan(result.iloc[0].rank_in_cohort_set))

    def test_label_conflict_and_leakage_rejected(self):
        a, b = self.run_fixture(), self.run_fixture('chandra', 'multimodal')
        b.splits['train'] = ['a']
        with self.assertRaisesRegex(ValueError, 'leakage'):
            paired_delta(a, b, n_boot=0)
        b.splits['train'] = []
        f = pd.read_csv(b.path/'test_predictions.csv')
        f.loc[0, 'label'] = 1
        f.to_csv(b.path/'test_predictions.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'Mismatch'):
            paired_delta(a, b, n_boot=0)

    def test_discovery_corruption_and_incomplete(self):
        a = self.run_fixture()
        (a.path/'config.json').write_text(json.dumps(a.metadata))
        (self.root/'broken').mkdir()
        (self.root/'broken/config.json').write_text('{')
        runs, inventory, issues = discover([self.root])
        self.assertEqual(len(runs), 1)
        self.assertEqual(inventory.status.iloc[0], 'incomplete')
        self.assertEqual(len(issues), 1)
        table, issues, _ = comparison(list(runs.values()))
        self.assertTrue(table.empty)
        self.assertEqual(len(issues), 1)


if __name__ == '__main__':
    unittest.main()
