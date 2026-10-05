import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from clusterprep.config_wizard import create_config
from clusterprep.training import ExperimentConfig
from clusterprep.__main__ import main


class ConfigWizardTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "training.json"
        self.messages = []

    def test_defaults_are_current_training_defaults(self):
        result = create_config(self.path, input_fn=lambda _: "", print_fn=self.messages.append)
        self.assertEqual(result, self.path.resolve())
        spec = json.loads(self.path.read_text())
        config = ExperimentConfig(**spec["experiments"][0])
        self.assertEqual(config.learning_rate, ExperimentConfig("default").learning_rate)
        self.assertEqual(config.device, ExperimentConfig("default").device)
        self.assertTrue(Path(spec["data_dir"]).is_absolute())
        self.assertEqual(spec["comparison_cohort"], "all")

    def test_invalid_input_multiple_modes_and_folds(self):
        replies = {
            "Nombre de folds": iter(["1", "5"]),
            "Index du fold": iter(["8", "2"]),
            "Fraction validation": iter(["0.9", "0.2"]),
            "Mode (": iter(["pair", "multimodal"]),
            "Modèle [": iter(["fusion", "cnn"]),
            "Taille des lots": iter(["z", "0", "4", "4"]),
            "Learning rate": iter(["nan", "0.001", "0.002"]),
            "Ajouter une autre": iter(["oui", "non"]),
            "Device (": iter(["cpu", "cpu"]),
            "Cache FITS": iter(["aucun", "aucun"]),
        }
        def read(prompt):
            return next((next(values) for prefix, values in replies.items() if prompt.startswith(prefix)), "")
        create_config(self.path, input_fn=read, print_fn=self.messages.append)
        spec = json.loads(self.path.read_text())
        self.assertEqual(spec["split"]["n_folds"], 5)
        self.assertEqual(spec["split"]["fold_index"], 2)
        self.assertEqual(spec["comparison_cohort"], "paired")
        self.assertEqual(len(spec["experiments"]), 2)
        self.assertEqual(spec["experiments"][0]["learning_rate"], .001)
        self.assertIsNone(spec["experiments"][0]["processed_dir"])
        self.assertTrue(any("À corriger" in m for m in self.messages))

    def test_existing_file_preserved_and_cancellation(self):
        self.path.write_text("unchanged")
        self.assertIsNone(create_config(self.path, input_fn=lambda _: "", print_fn=self.messages.append))
        self.assertEqual(self.path.read_text(), "unchanged")
        def cancel(_):
            raise EOFError
        self.assertIsNone(create_config(self.path, input_fn=cancel, print_fn=self.messages.append))
        self.assertEqual(self.path.read_text(), "unchanged")

    def test_cli_dispatch_and_model_parameters(self):
        def read(prompt):
            if prompt.startswith("Modèle ["):
                return "image_cnn"
            if prompt.startswith("Paramètres du modèle"):
                return '{"hidden_dim": 12, "classifier_hidden_dim": 8}'
            return ""
        with patch("builtins.input", side_effect=read), patch("builtins.print"):
            main(["createconfig", "--output", str(self.path)])
        config = json.loads(self.path.read_text())["experiments"][0]
        self.assertEqual(config["model"], "image_cnn")
        self.assertEqual(config["model_params"]["hidden_dim"], 12)
