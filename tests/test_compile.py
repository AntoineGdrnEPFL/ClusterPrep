"""Compilation preserves training updates, model identity and checkpoint format."""
import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from clusterprep import Trainer, TrainingConfig
from clusterprep.training_core import compile_training_model


@pytest.mark.parametrize('options', [
    {'use_compile': 'yes'}, {'use_compile': 1},
    {'compile_dynamic': 1}, {'compile_backend': ''}, {'compile_mode': 'fast'},
])
def test_invalid_compile_options(options):
    with pytest.raises(ValueError):
        TrainingConfig(**options)


def test_device_policy_and_single_compilation():
    model = nn.Linear(2, 2)
    config = TrainingConfig(use_scheduler=False)
    with patch('torch.compile', return_value=model) as compile_mock:
        assert compile_training_model(model, config, 'cpu') is model
        assert compile_training_model(model, replace(config, use_compile=False), 'mps') is model
        assert compile_training_model(model, replace(config, use_compile=False), 'cuda') is model
        compile_mock.assert_not_called()
        compile_training_model(model, config, 'cuda:0')
        compile_mock.assert_called_once_with(
            model, backend='inductor', mode='default', dynamic=None, fullgraph=False)
        compile_mock.reset_mock()
        compile_training_model(model, config, 'mps')
        compile_mock.assert_called_once_with(
            model, backend='inductor', mode='default', dynamic=None, fullgraph=False)
    with patch('torch.compile', None), pytest.raises(RuntimeError, match='PyTorch'):
        compile_training_model(model, replace(config, use_compile=True), 'cpu')


@pytest.mark.parametrize('inputs', ['tensor', 'tuple', 'dict'])
def test_real_compiled_backward_variable_batches_and_checkpoint(inputs):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(2, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, x, extra=None):
            return self.head(self.encoder(x if extra is None else x + extra).relu())

    def adapt(batch):
        x = batch['raw']
        return x if inputs == 'tensor' else ((x, x) if inputs == 'tuple' else {'x': x, 'extra': x})

    torch.manual_seed(7)
    base = Model()
    config = TrainingConfig(epochs=1, use_scheduler=False, use_mixup=False, use_compile=False)
    loader = DataLoader([
        {'raw': torch.tensor([float(i), 1.]), 'label': torch.tensor(i % 2)}
        for i in range(5)
    ], batch_size=2)
    eager = Trainer(copy.deepcopy(base), config, ['a', 'b'], [3, 2],
                    steps_per_epoch=3, input_adapter=adapt)
    original = copy.deepcopy(base)
    with patch('torch.compile', wraps=torch.compile) as compile_mock:
        compiled = Trainer(original, replace(config, use_compile=True, compile_backend='aot_eager'),
                           ['a', 'b'], [3, 2], steps_per_epoch=3, input_adapter=adapt)
        assert compiled.model is original
        # Reuse the callable across freeze/unfreeze, as in transfer learning.
        for frozen in (True, False):
            for trainer in (eager, compiled):
                trainer.model.encoder.requires_grad_(not frozen)
                trainer.fit(loader)
            for key, expected in eager.model.state_dict().items():
                torch.testing.assert_close(compiled.model.state_dict()[key], expected)
        compile_mock.assert_called_once()
    state = compiled._checkpoint(1)['model_state_dict']
    assert not any('_orig_mod' in key for key in state)
    restored = Model()
    restored.load_state_dict(state, strict=True)
    # Validation/prediction must not compile a second graph.
    with patch.object(compiled.training_model, 'forward', side_effect=AssertionError('compiled eval')):
        scores, predictions = compiled.evaluate(loader)
        assert scores['n_samples'] == len(predictions) == 5
        torch.testing.assert_close(compiled.predict(loader), eager.predict(loader))


def test_compiled_multilabel_masked_mixup():
    from clusterprep.training import ExperimentConfig

    torch.manual_seed(11)
    base = nn.Linear(2, 2)
    config = ExperimentConfig(name='compiled_rr_rh', task='rr_rh', epochs=1,
                              use_scheduler=False, mixup_probability=1, use_compile=False)
    loader = DataLoader([
        {'raw': torch.tensor([float(i), 1.]),
         'label': torch.tensor([float(i % 2), float(i // 2)]),
         'label_mask': torch.tensor([1., float(i != 0)])}
        for i in range(4)
    ], batch_size=4)
    trainers = [Trainer(copy.deepcopy(base), c, ['RR', 'RH'], [2, 2],
                        n_train=[4, 3], steps_per_epoch=1)
                for c in (config, replace(config, use_compile=True, compile_backend='aot_eager'))]
    results = []
    for trainer in trainers:
        with patch('numpy.random.beta', return_value=.25), \
             patch('torch.randperm', return_value=torch.tensor([3, 2, 1, 0])):
            results.append(trainer.train_one_epoch(loader)[0])
    assert results[0]['loss'] == pytest.approx(results[1]['loss'], abs=1e-6)
    for key, expected in trainers[0].model.state_dict().items():
        torch.testing.assert_close(trainers[1].model.state_dict()[key], expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS requis')
def test_mps_inductor_cnn_training_updates():
    from clusterprep.models import SimpleCNN

    torch.manual_seed(42)
    original = SimpleCNN(width=4).to('mps')
    compiled_model = copy.deepcopy(original)
    compiled = compile_training_model(compiled_model, TrainingConfig(), 'mps')
    assert compiled is not compiled_model
    x = torch.randn(4, 1, 32, 32, device='mps')
    y = torch.tensor([0, 1, 0, 1], device='mps')
    for model, forward in ((original, original), (compiled_model, compiled)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(forward(x), y)
            loss.backward()
            optimizer.step()
    torch.mps.synchronize()
    for key, expected in original.state_dict().items():
        torch.testing.assert_close(compiled_model.state_dict()[key], expected, rtol=1e-4, atol=1e-5)
