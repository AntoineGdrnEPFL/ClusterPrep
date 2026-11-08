"""Explicit radio/optical layouts retain the DualEncoderCNN architecture."""
from dataclasses import replace

import pytest
import torch

from clusterprep.models import DualEncoderCNN, build_model
from clusterprep.training import ExperimentConfig


@pytest.mark.parametrize('modalities,channels', [(['radio'], 1), (['radio', 'legacy'], 7)])
def test_radio_optical_forward_backward_and_transfer(modalities, channels):
    config = ExperimentConfig(name='optical', mode='multimodal', model='dual_encoder_cnn',
                              modalities=modalities, size=32)
    model = build_model(config, 2)
    assert isinstance(model, DualEncoderCNN)
    assert model.chandra_encoder is None
    x = torch.randn(2, channels, 32, 32)
    if channels == 7:
        x[:, 4:] = 1
    logits = model(x)
    assert logits.shape == (2, 2)
    logits.sum().backward()
    assert model.radio_encoder.encoder[0].weight.grad is not None
    if channels == 7:
        assert model.legacy_encoder.encoder[0].weight.grad is not None
        assert set(model.transfer_modules()) == {'radio', 'legacy', 'head'}
        model.eval()
        x[:, 4:] = 0
        changed = x.clone()
        changed[:, 1:4] += 10
        with torch.no_grad():
            torch.testing.assert_close(model(x), model(changed))


@pytest.mark.parametrize('mode,modalities', [('raw', ['radio']), ('multimodal', ['radio', 'xray'])])
def test_historical_checkpoint_compatibility(mode, modalities):
    config = ExperimentConfig(name='compat', model='dual_encoder_cnn', mode=mode, size=32)
    historical = build_model(config, 2).eval()
    explicit = build_model(replace(config, modalities=modalities), 2).eval()
    explicit.load_state_dict(historical.state_dict(), strict=True)
    x = torch.rand(2, 1 if mode == 'raw' else 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(explicit(x), historical(x))


def test_three_modalities_rejected():
    config = ExperimentConfig(name='invalid', model='dual_encoder_cnn',
                              modalities=['radio', 'xray', 'legacy'])
    with pytest.raises(ValueError, match='DualEncoderCNN'):
        build_model(config, 2)
