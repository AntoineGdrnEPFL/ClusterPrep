from dataclasses import replace
import importlib.util
import unittest
import torch
from torch import nn
from clusterprep import ExperimentConfig, ImageCNN, build_model, register_model, list_models
from clusterprep.training import _MergeSingletonBatchSampler


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_image_cnn_modes_and_constructor_batchnorm(self):
        for mode in ("raw", "pair", "multimodal"):
            config = ExperimentConfig("image", model="image_cnn", mode=mode, size=16,
                                      model_params={"hidden_dim": 7, "classifier_hidden_dim": 5})
            model = build_model(config, 3)
            self.assertIsInstance(model, ImageCNN)
            self.assertEqual(model.classifier[0].out_features, 7)
            for layer in model.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    self.assertEqual(layer.num_batches_tracked.item(), 0)
            x = torch.randn(2, 1 if mode == "raw" else 3, 16, 16)
            output = model(x)
            self.assertEqual(tuple(output.shape), (2, 3))
            output.sum().backward()
            model.eval()
            with torch.no_grad():
                self.assertEqual(tuple(model(x[:1]).shape), (1, 3))

    def test_registry_parameters_and_import(self):
        @register_model("test_custom")
        class Custom(nn.Module):
            def __init__(self, in_channels, num_classes, features=4):
                super().__init__()
                self.head = nn.Linear(in_channels, num_classes)
                self.features = features
            def forward(self, x):
                return self.head(x.mean((2, 3)))
        config = ExperimentConfig("custom", model="test_custom", model_params={"features": 9})
        self.assertEqual(build_model(config, 2).features, 9)
        self.assertIn("test_custom", list_models())
        with self.assertRaisesRegex(ValueError, "déjà enregistré"):
            register_model("test_custom")(Custom)
        imported = build_model(replace(config, model="clusterprep.models:ImageCNN",
                                       size=16, model_params={}), 2)
        self.assertIsInstance(imported, ImageCNN)
        for changes in ({"model": "absent"}, {"model": "fusion", "mode": "raw"},
                        {"model_params": {"typo": 1}}, {"model_params": {"in_channels": 6}}):
            with self.assertRaises(ValueError):
                build_model(replace(config, **changes), 2)

    def test_dual_encoder_modes_and_checkpoint(self):
        self.assertEqual(list_models()["dual_encoder_cnn"], ["multimodal", "pair", "raw"])
        for mode in ("raw", "pair", "multimodal"):
            for use_mask, use_coverage in ((True, True), (False, True), (True, False), (False, False)):
                with self.subTest(mode=mode, use_mask=use_mask, use_coverage=use_coverage):
                    config = ExperimentConfig(
                        "dual", model="dual_encoder_cnn", mode=mode, size=16,
                        model_params={"use_mask": use_mask, "use_coverage": use_coverage})
                    model = build_model(config, 3)
                    channels = 1 if mode == "raw" else 3
                    x = torch.randn(2, channels, 16, 16)
                    if mode != "raw":
                        x[:, 2] = 1
                        x[0, 2] = 0
                    else:
                        self.assertIsNone(model.chandra_encoder)
                        self.assertFalse(model.use_coverage)
                        self.assertEqual(model.classifier[0].in_features, model.radio_encoder.output_dim)
                        self.assertFalse(any(k.startswith("chandra_encoder.") for k in model.state_dict()))
                    output = model(x)
                    self.assertEqual(tuple(output.shape), (2, 3))
                    self.assertTrue(torch.isfinite(output).all())
                    output.square().sum().backward()
                    self.assertIsNotNone(model.radio_encoder.encoder[0].weight.grad)
                    if mode != "raw":
                        self.assertIsNotNone(model.chandra_encoder.encoder[0].weight.grad)
                    model.eval()
                    restored = build_model(config, 3)
                    restored.load_state_dict(model.state_dict())
                    restored.eval()
                    with torch.no_grad():
                        torch.testing.assert_close(restored(x[:1]), model(x[:1]))
                        if mode != "raw":
                            changed = x[:1].clone()
                            changed[:, 1] += 100
                            torch.testing.assert_close(model(changed), model(x[:1]))

    def test_singleton_batch_merged_without_lost_samples(self):
        for count in (2, 3, 4, 5, 8, 9):
            sampler = _MergeSingletonBatchSampler(range(count), 4, drop_last=False)
            batches = list(sampler)
            self.assertEqual(len(batches), len(sampler))
            self.assertEqual([i for batch in batches for i in batch], list(range(count)))
            self.assertTrue(all(len(b) > 1 for b in batches))

    @unittest.skipUnless(importlib.util.find_spec("kymatio"), "kymatio optionnel absent")
    def test_dual_ssn_modes_gradients_and_checkpoint(self):
        for name in ("dual_ssn", "dual_encoder_ssn"):
            self.assertEqual(list_models()[name], ["multimodal", "pair", "raw"])
            for mode in ("raw", "pair", "multimodal"):
                with self.subTest(model=name, mode=mode):
                    config = ExperimentConfig("ssn", model=name, mode=mode, size=16,
                                              model_params={"L": 2, "hidden_dim2": 4})
                    model = build_model(config, 3)
                    for layer in model.modules():
                        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                            self.assertEqual(layer.num_batches_tracked.item(), 0)
                    x = torch.rand(2, 1 if mode == "raw" else 3, 16, 16)
                    if mode != "raw":
                        x[:, 2] = 1
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    output = model(x)
                    self.assertEqual(tuple(output.shape), (2, 3))
                    nn.functional.cross_entropy(output, torch.tensor([0, 2])).backward()
                    encoders = ([model.encoder] if name == "dual_ssn" else
                                [model.radio_encoder] + ([model.chandra_encoder] if mode != "raw" else []))
                    for encoder in encoders:
                        for branch in (encoder.cnn_encoder, encoder.conv_to_latent_scat):
                            grad = branch[0][0].weight.grad
                            self.assertTrue(torch.isfinite(grad).all())
                            self.assertGreater(grad.abs().sum().item(), 0)
                    optimizer.step()
                    model.eval()
                    restored = build_model(config, 3)
                    restored.load_state_dict(model.state_dict())
                    restored.eval()
                    with torch.no_grad():
                        torch.testing.assert_close(restored(x[:1]), model(x[:1]))

    @unittest.skipUnless(importlib.util.find_spec("kymatio"), "kymatio optionnel absent")
    def test_dual_ssn_scattering_scales_and_multichannel(self):
        from clusterprep.models import DualSSNEncoder
        for J in (1, 2, 3, 4):
            for order in (1, 2):
                with self.subTest(J=J, max_order=order):
                    encoder = DualSSNEncoder((3, 16, 16), hidden_dim2=2, J=J, L=2, max_order=order)
                    x = torch.rand(2, 3, 16, 16)
                    scat = encoder.scattering(x)
                    for channel in range(3):
                        torch.testing.assert_close(scat[:, channel], encoder.scattering(x[:, channel].contiguous()))
                    self.assertEqual(tuple(encoder(x).shape), (2, encoder.output_dim))
        for params in ({"J": 0}, {"J": 5}, {"L": 0}, {"max_order": 3}, {"hidden_dim2": 0}):
            with self.assertRaises(ValueError):
                DualSSNEncoder((1, 16, 16), **params)
        with self.assertRaises(ValueError):
            DualSSNEncoder((1, 8, 8), J=4)

    @unittest.skipUnless(importlib.util.find_spec("kymatio"), "kymatio optionnel absent")
    def test_dual_encoder_ssn_missing_chandra_and_options(self):
        from clusterprep import DualEncoderSSN
        for use_mask in (False, True):
            for use_coverage in (False, True):
                model = DualEncoderSSN((3, 16, 16), L=2, hidden_dim2=2,
                                       use_mask=use_mask, use_coverage=use_coverage)
                x = torch.rand(2, 3, 16, 16)
                x[:, 2] = 0
                model(x).sum().backward()
                for param in model.chandra_encoder.parameters():
                    self.assertEqual(param.grad.abs().sum().item(), 0)
                model.eval()
                changed = x.clone()
                changed[:, 1] += 100
                with torch.no_grad():
                    torch.testing.assert_close(model(x), model(changed))
        model = DualEncoderSSN((1, 16, 16), L=2)
        self.assertIsNone(model.chandra_encoder)
        self.assertFalse(model.use_coverage)
        with self.assertRaises(ValueError):
            DualEncoderSSN((2, 16, 16))
