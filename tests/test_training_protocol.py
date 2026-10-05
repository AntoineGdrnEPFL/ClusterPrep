"""Régressions du protocole legacy et des adaptateurs multimodaux."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.data import DataLoader
from clusterprep import (TrainingConfig, Trainer, EarlyStopping, build_criterion,
    compute_class_weights, build_optimizer, build_scheduler, mixup_batch)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def data(self):
        return [{"raw": torch.tensor([float(i), 1.]), "label": torch.tensor(i % 2)} for i in range(5)]

    def test_criterion_matches_pytorch_and_ablations(self):
        config = TrainingConfig()
        torch.testing.assert_close(compute_class_weights([2, 6]), torch.tensor([4., 4/3]))
        logits = torch.tensor([[2., 1.], [0., 4.], [1., 2.]])
        y = torch.tensor([0, 1, 1])
        criterion = build_criterion(config, [2, 6])
        expected = nn.CrossEntropyLoss(weight=torch.tensor([4., 4/3]), label_smoothing=.1)
        torch.testing.assert_close(criterion(logits, y), expected(logits, y))
        plain = build_criterion(replace(config, use_class_weights=False, label_smoothing=0), [2, 6])
        torch.testing.assert_close(plain(logits, y), nn.CrossEntropyLoss()(logits, y))
        optimizer = build_optimizer(nn.Linear(2, 2), config)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]['lr'], 5e-5)
        self.assertIsNone(build_scheduler(optimizer, replace(config, use_scheduler=False), 2))
        with self.assertRaisesRegex(ValueError, 'Aucun batch'):
            build_scheduler(optimizer, config, 0)

    def test_mixup_aligns_all_modalities_and_mask(self):
        x = torch.arange(4.).reshape(4, 1)
        y = torch.arange(4)
        permutation = torch.tensor([3, 2, 0, 1])
        with patch('numpy.random.beta', return_value=.25), patch('torch.randperm', return_value=permutation):
            mixed, a, b, lam = mixup_batch({'radio': x, 'xray': (x * 10, x.bool())}, y)
        torch.testing.assert_close(mixed['radio'], .25 * x + .75 * x[permutation])
        torch.testing.assert_close(mixed['xray'][0], mixed['radio'] * 10)
        torch.testing.assert_close(mixed['xray'][1], .25 * x.bool() + .75 * x.bool()[permutation])
        torch.testing.assert_close(a, y)
        torch.testing.assert_close(b, y[permutation])
        self.assertEqual(lam, .25)

    def test_evaluation_inference_and_sample_weighted_loss(self):
        class Probe(nn.Linear):
            def forward(self, x):
                self.asserted_inference = torch.is_inference_mode_enabled() and not self.training
                return super().forward(x)
        model = Probe(2, 2)
        config = TrainingConfig(epochs=2, batch_size=2)
        loader = DataLoader(self.data(), batch_size=2)
        trainer = Trainer(model, config, ['a', 'b'], [3, 2], steps_per_epoch=3,
                          input_adapter=lambda b: b['raw'])
        with patch('clusterprep.training.mixup_batch', side_effect=AssertionError('MixUp en validation')):
            actual, _ = trainer.evaluate(loader)
        with torch.inference_mode():
            expected = sum(trainer.criterion(model(b['raw']), b['label']).item() * len(b['label'])
                           for b in loader) / 5
        self.assertTrue(model.asserted_inference)
        self.assertAlmostEqual(actual['loss'], expected)
        self.assertEqual(actual['n_samples'], 5)
        self.assertEqual(trainer.predict(loader).shape, (5, 2))

    def test_batch_step_order_and_mixup_loss(self):
        config = TrainingConfig(epochs=2, mixup_probability=1)
        model = nn.Linear(2, 2)
        trainer = Trainer(model, config, ['a', 'b'], [3, 2], steps_per_epoch=3,
                          input_adapter=lambda b: b['raw'])
        events = []
        optimizer_step = trainer.optimizer.step
        scheduler_step = trainer.scheduler.step
        clip = nn.utils.clip_grad_norm_
        def optimizer(*args, **kwargs):
            events.append('optimizer')
            return optimizer_step(*args, **kwargs)
        def scheduler(*args, **kwargs):
            events.append('scheduler')
            return scheduler_step(*args, **kwargs)
        def clipping(*args, **kwargs):
            events.append('clip')
            return clip(*args, **kwargs)
        with patch.object(trainer.optimizer, 'step', side_effect=optimizer), \
             patch.object(trainer.scheduler, 'step', side_effect=scheduler), \
             patch('torch.nn.utils.clip_grad_norm_', side_effect=clipping), \
             patch('clusterprep.training.mixup_batch', wraps=mixup_batch) as mixing:
            trainer.train_one_epoch(DataLoader(self.data(), batch_size=2))
        self.assertEqual(events, ['clip', 'optimizer', 'scheduler'] * 3)
        self.assertEqual(mixing.call_count, 3)
        self.assertEqual(trainer.scheduler.last_epoch, 3)

    def test_restore_is_deep_copy_and_fit_without_early_stop(self):
        model = nn.Linear(2, 2)
        initial = model.weight.detach().clone()
        stopping = EarlyStopping(patience=1)
        stopping.update(1., model)
        with torch.no_grad():
            model.weight.add_(10)
        self.assertEqual(stopping.update(1., model), (False, True))
        stopping.restore(model)
        torch.testing.assert_close(model.weight, initial)
        config = TrainingConfig(epochs=3, use_scheduler=False, use_early_stopping=False, patience=1)
        trainer = Trainer(model, config, ['a', 'b'], [3, 2], steps_per_epoch=1)
        states = []
        def train(_):
            with torch.no_grad():
                model.weight.add_(1)
            states.append(model.weight.detach().clone())
            return {'loss': 1.}, []
        with patch.object(trainer, 'train_one_epoch', side_effect=train), \
             patch.object(trainer, 'evaluate', side_effect=[({'loss': v}, []) for v in (1., 2., 3.)]):
            history = trainer.fit(None, None)
        self.assertEqual(len(history), 3)
        self.assertEqual(trainer.best_epoch, 1)
        torch.testing.assert_close(model.weight, states[0])

    def test_invalid_config(self):
        for kwargs in ({'mixup_probability': 1.1}, {'label_smoothing': -1},
                       {'warmup_fraction': 0}, {'mixup_alpha': 0}, {'gradient_clip_norm': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingConfig(**kwargs)

    def test_exact_mixup_loss_and_disabled_methods(self):
        config = TrainingConfig(epochs=1, use_scheduler=False, mixup_probability=1,
                                gradient_clip_norm=0)
        model = nn.Linear(2, 2)
        trainer = Trainer(model, config, ['a', 'b'], [3, 2], steps_per_epoch=1,
                          input_adapter=lambda b: b['raw'])
        loader = DataLoader(self.data(), batch_size=5)
        batch = next(iter(loader))
        permutation = torch.tensor([4, 3, 2, 1, 0])
        x, y = batch['raw'], batch['label']
        with torch.no_grad():
            logits = model(.25 * x + .75 * x[permutation])
            expected = .25 * trainer.criterion(logits, y) + .75 * trainer.criterion(logits, y[permutation])
        with patch('numpy.random.beta', return_value=.25), \
             patch('torch.randperm', return_value=permutation), \
             patch('torch.nn.utils.clip_grad_norm_', side_effect=AssertionError('Clipping désactivé')):
            result, _ = trainer.train_one_epoch(loader)
        self.assertAlmostEqual(result['loss'], expected.item(), places=6)
        trainer.config = replace(config, use_mixup=False)
        with patch('clusterprep.training.mixup_batch', side_effect=AssertionError('MixUp désactivé')):
            trainer.train_one_epoch(loader)
