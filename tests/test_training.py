import csv
import contextlib
import io
import importlib.util
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from clusterprep import (ClusterDataset, ExperimentConfig, EarlyStopping, FusionCNN,
    classification_metrics, run_experiments, run_experiment, compare_experiments, build_model,
    plot_history)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [180, 45]
        wcs.wcs.crpix = [4, 4]
        wcs.wcs.cdelt = [-.01, .01]
        self.info = {}
        rng = np.random.default_rng(4)
        for i in range(12):
            path = self.root / f"{i}.fits"
            fits.PrimaryHDU(rng.normal(size=(8, 8)).astype(np.float32), wcs.to_header()).writeto(path)
            self.info[str(i)] = {"cluster_id": str(i), "raw_path": str(path),
                "chandra_path": str(path) if i % 4 < 2 else None,
                "class": "DE" if i % 2 == 0 else "NDE"}
        self.splits = ([str(i) for i in range(4)], [str(i) for i in range(4, 8)],
                       [str(i) for i in range(8, 12)])

    def test_early_stopping(self):
        stopping = EarlyStopping(patience=2, min_delta=.1)
        self.assertEqual(stopping.step(1), (True, False))
        self.assertEqual(stopping.step(.95), (False, False))
        self.assertEqual(stopping.step(.8), (True, False))
        self.assertEqual(stopping.step(.85), (False, False))
        self.assertEqual(stopping.step(.9), (False, True))
        stopping = EarlyStopping(patience=1, mode="max")
        self.assertEqual(stopping.step(.5), (True, False))
        self.assertEqual(stopping.step(.6), (True, False))
        self.assertEqual(stopping.step(.6), (False, True))
        with self.assertRaises(ValueError):
            stopping.step(float("nan"))

    def test_metrics_missing_class_and_class_order(self):
        metrics = classification_metrics([0, 0], [[.9, .1], [.8, .2]], ["DE", "NDE"])
        self.assertIsNone(metrics["roc_auc_macro"])
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 0]])
        json.dumps(metrics, allow_nan=False)
        metrics = classification_metrics([0, 1], [[.9, .1], [.1, .9]], ["DE", "NDE"])
        self.assertEqual(metrics["roc_auc_macro"], 1.)

    def test_missing_modality_cache_and_fusion(self):
        ds = ClusterDataset(self.info, ["2"], ["DE", "NDE"], mode="multimodal", size=8, cache=True, processed_dir=self.root / "processed")
        first = ds[0]
        self.assertFalse(first["has_chandra"])
        self.assertEqual(first["chandra"].sum(), 0)
        self.assertEqual(first["chandra_mask"].sum(), 0)
        first["raw"].zero_()
        self.assertGreater(ds[0]["raw"].sum(), 0)
        model = FusionCNN(width=2)
        model(torch.cat([ds[0][k] for k in ("raw", "chandra", "chandra_mask")])[None]).sum().backward()
        for param in model.chandra.parameters():
            self.assertEqual(param.grad.abs().sum(), 0)

    def test_experiments_checkpoint_predictions_and_comparison(self):
        base = ExperimentConfig("radio", size=8, width=2, batch_size=2,
            device="cpu", epochs=5, augment=False, patience=1, min_delta=100,
            processed_dir=str(self.root / "processed"), verbose=False)
        configs = [base, replace(base, name="pair", mode="pair", class_weight="balanced"),
                   replace(base, name="both", mode="multimodal", model="fusion")]
        runs = run_experiments(self.info, configs, self.root / "runs", splits=self.splits)
        for path, config in zip(runs, configs):
            result = json.loads((path / "metrics.json").read_text())
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["epochs_ran"], 2)
            self.assertTrue(result["stopped_early"])
            self.assertEqual(len((path / "history.jsonl").read_text().splitlines()), 2)
            checkpoint = torch.load(path / "best.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            model = build_model(config, 2)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            with (path / "test_predictions.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            ds = ClusterDataset(self.info, [row["key"] for row in rows], ["DE", "NDE"],
                                mode=config.mode, size=8, processed_dir=self.root / "processed")
            for row, sample in zip(rows, ds):
                names = ("raw",) if config.mode == "raw" else ("raw", "chandra", "chandra_mask")
                with torch.no_grad():
                    proba = model(torch.cat([sample[k] for k in names])[None]).softmax(1)[0]
                np.testing.assert_allclose(proba, [float(row["probability_DE"]),
                                                  float(row["probability_NDE"])], atol=1e-6)
        table = compare_experiments(runs, self.root / "comparison.csv")
        self.assertEqual(len({r["cohort_id"] for r in table}), 1)
        self.assertTrue(all(r["n_samples"] == 2 for r in table))
        figure = plot_history(runs[0], self.root / "history.png")
        plt.close(figure)
        self.assertGreater((self.root / "history.png").stat().st_size, 1000)
        with self.assertRaisesRegex(ValueError, "Cohortes"):
            compare_experiments(runs, cohort="all")
        with contextlib.redirect_stderr(io.StringIO()) as console:
            repeated = run_experiment(self.info, replace(base, verbose=True),
                                      self.root / "runs", splits=self.splits)
        self.assertIn("Époque 001/5", console.getvalue())
        self.assertIn("Arrêt anticipé", console.getvalue())
        self.assertIn("12 réutilisés", console.getvalue())
        self.assertNotIn("\r", console.getvalue())
        self.assertNotEqual(repeated, runs[0])
        self.assertEqual((repeated / "test_predictions.csv").read_text(),
                         (runs[0] / "test_predictions.csv").read_text())

    def test_leakage_and_missing_train_class_rejected(self):
        bad = (self.splits[0], self.splits[0], self.splits[2])
        with self.assertRaisesRegex(ValueError, "Fuite"):
            run_experiment(self.info, ExperimentConfig("bad"), self.root, splits=bad)
        self.info["1"]["chandra_path"] = None
        with self.assertRaisesRegex(ValueError, "Chaque classe"):
            run_experiment(self.info, ExperimentConfig("bad", mode="pair"), self.root, splits=self.splits)

    def test_image_cnn_training_and_checkpoint_with_singleton_remainder(self):
        config = ExperimentConfig("image", model="image_cnn", size=8, batch_size=3,
            device="cpu", drop_last_train=False, epochs=1, augment=False, verbose=False, processed_dir=str(self.root / "processed"),
            model_params={"hidden_dim": 5, "classifier_hidden_dim": 4})
        run = run_experiment(self.info, config, self.root / "runs", splits=self.splits)
        result = json.loads((run / "metrics.json").read_text())
        self.assertEqual(result["test"]["n_samples"], 4)
        history = json.loads((run / "history.jsonl").read_text())
        self.assertEqual(history["train"]["n_samples"], 4)
        checkpoint = torch.load(run / "best.pt", weights_only=True)
        restored = build_model(ExperimentConfig(**checkpoint["config"]), 2)
        restored.load_state_dict(checkpoint["model_state_dict"])
        self.assertEqual(restored.classifier[0].out_features, 5)

    def test_drop_last_keeps_validation_and_test_complete(self):
        config = ExperimentConfig("drop", model="image_cnn", size=8, batch_size=3,
            device="cpu", epochs=1, augment=False, verbose=False,
            processed_dir=str(self.root / "processed"),
            model_params={"hidden_dim": 5, "classifier_hidden_dim": 4})
        run = run_experiment(self.info, config, self.root / "runs", splits=self.splits)
        history = json.loads((run / "history.jsonl").read_text())
        result = json.loads((run / "metrics.json").read_text())
        self.assertEqual(history["train"]["n_samples"], 3)
        self.assertEqual(result["validation"]["n_samples"], 4)
        self.assertEqual(result["test"]["n_samples"], 4)

    @unittest.skipUnless(importlib.util.find_spec("kymatio"), "kymatio optionnel absent")
    def test_dual_ssn_training_and_checkpoint(self):
        for name in ("dual_ssn", "dual_encoder_ssn"):
            with self.subTest(model=name):
                config = ExperimentConfig(name, model=name, mode="multimodal", size=8,
                    batch_size=3, drop_last_train=False, device="cpu", epochs=1,
                    augment=False, verbose=False, processed_dir=str(self.root / "processed"),
                    model_params={"L": 2, "hidden_dim2": 2})
                run = run_experiment(self.info, config, self.root / "runs", splits=self.splits)
                history = json.loads((run / "history.jsonl").read_text())
                self.assertEqual(history["train"]["n_samples"], 4)
                result = json.loads((run / "metrics.json").read_text())
                self.assertEqual(result["test"]["n_samples"], 4)
                checkpoint = torch.load(run / "best.pt", weights_only=True)
                restored = build_model(ExperimentConfig(**checkpoint["config"]), 2)
                restored.load_state_dict(checkpoint["model_state_dict"])
                restored.eval()
                self.assertTrue(torch.isfinite(restored(torch.rand(1, 3, 8, 8))).all())



if __name__ == "__main__":
    unittest.main()
