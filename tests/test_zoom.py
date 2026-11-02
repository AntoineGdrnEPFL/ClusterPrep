"""Zoom regression tests on synthetic, WCS-aligned astronomical images."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from clusterprep import ClusterDataset, ExperimentConfig, plot_zoom_variants
from clusterprep.augmentation import transform_aligned, validate_zoom, variant_count
from clusterprep.scattering import prepare_scattering, load_scattering, scattering_paths
from clusterprep.training import run_experiment
from test_audit import catalog, single_thread


def zoom(factors=(.9, 1., 1.1), **kwargs):
    return dict(enabled=True, factors=list(factors), **kwargs)


@pytest.mark.parametrize('bad', [[], {'typo': 1}, {'enabled': 1},
    {'enabled': True}, zoom(()), zoom((0,)), zoom((-1,)), zoom((float('nan'),)),
    zoom((float('inf'),)), zoom((True,)), zoom(('1',)),
    zoom(interpolation='cubic'), zoom(padding_mode='noise'), zoom(fill_value=float('inf'))])
def test_invalid_config(bad):
    with pytest.raises(ValueError, match='zoom'):
        ExperimentConfig(name='invalid', augmentation={'zoom': bad})


def test_old_json_and_example_configs():
    assert validate_zoom()['enabled'] is False
    assert ExperimentConfig(**json.loads('{"name":"old"}')).augmentation == {}
    assert variant_count(zoom((.9, 1, 1.1, 1.2))) == 96
    spec = json.loads((Path(__file__).parents[1] / 'configs/zoom/rr_rh.json').read_text())
    configs = [ExperimentConfig(**e) for e in spec['experiments']]
    assert [variant_count(c.augmentation['zoom']) for c in configs] == [24, 72, 120]
    assert len({json.dumps({k: v for k, v in e.items() if k not in {'name', 'augmentation'}}, sort_keys=True)
                for e in spec['experiments']}) == 1


@pytest.mark.parametrize('mode', ['raw', 'pair', 'multimodal'])
@pytest.mark.parametrize('task', ['de_nde', 'rr_rh', 'rr_binary', 'rh_binary'])
def test_index_identity_reproducibility(catalog, mode, task):
    args = (catalog, ['c1.fits', 'c2.fits'], ['DE', 'NDE'])
    opts = dict(mode=mode, task=task, size=16, augment=True, cache=True, processed_dir=None)
    legacy = ClusterDataset(*args, **opts)
    disabled = ClusterDataset(*args, **opts, zoom={'enabled': False, 'factors': [.8, 1.2]})
    ds = ClusterDataset(*args, **opts, zoom=zoom())
    assert len(ds) == 144 and len(disabled) == 48
    names = ['raw'] if mode == 'raw' else ['raw', 'chandra', 'chandra_mask']
    for source in range(2):
        combinations = set()
        for i in range(72):
            index = source * 72 + i
            meta, sample = ds.get_item_info(index), ds[index]
            combinations.add((meta['rotation'], meta['flip_horizontal'], meta['zoom_factor']))
            assert meta['source_index'] == source and meta['variant_index'] == i
            assert sample['key'] == args[1][source]
            for name in names:
                assert sample[name].shape == (1, 16, 16)
                assert torch.isfinite(sample[name]).all()
                assert torch.equal(sample[name], ds[index][name])
                if 24 <= i < 48:
                    old_index = source * 24 + i - 24
                    assert torch.equal(sample[name], legacy[old_index][name])
                    assert torch.equal(disabled[old_index][name], legacy[old_index][name])
            assert torch.equal(sample['label'], legacy[source*24]['label'])
        assert len(combinations) == 72
    assert len(ds._cache) == 2  # only original images are held in memory
    with pytest.raises(IndexError):
        ds[len(ds)]


@pytest.mark.parametrize('padding', ['constant', 'reflect', 'replicate'])
def test_mask_weighted_interpolation_and_absence(padding):
    radio = torch.arange(81, dtype=torch.float32).reshape(1, 9, 9) / 80
    mask = torch.ones_like(radio)
    mask[:, 3:5, 4:6] = 0
    chandra = torch.where(mask.bool(), .7, float('nan'))
    source = torch.cat((radio, chandra, mask))
    options = zoom((.65,), padding_mode=padding, fill_value=.25)
    output = transform_aligned(source, angle=30, flip=True, scale=.65, zoom=options)
    assert torch.isfinite(output).all()
    assert set(output[2].unique().tolist()) == {0., 1.}
    assert output[2, 0].sum() == output[2, -1].sum() == 0
    torch.testing.assert_close(output[1][output[2].bool()], torch.full_like(output[1][output[2].bool()], .7))
    assert output[1][~output[2].bool()].count_nonzero() == 0
    if padding == 'constant':
        assert output[0, 0, 0].item() == pytest.approx(.25)
    source[1:] = 0
    assert transform_aligned(source, scale=.65, zoom=options)[1:].count_nonzero() == 0


@pytest.mark.parametrize('angle', [0, 30, 90, 210])
@pytest.mark.parametrize('flip', [False, True])
@pytest.mark.parametrize('scale', [.7, 1.3])
def test_common_grid_center_and_orientation(angle, flip, scale):
    # Asymmetric landmarks ensure rotation direction and flip order are tested.
    radio = torch.zeros(1, 31, 31)
    radio[0, 10:13, 19:22] = 1
    source = torch.cat((radio, radio.clone(), torch.ones_like(radio)))
    output = transform_aligned(source, angle, flip, scale, zoom((scale,)))
    valid = output[2].bool()
    torch.testing.assert_close(output[0][valid], output[1][valid])
    # Independent centre-of-mass oracle in pixel coordinates, y points down.
    radians = np.deg2rad(angle)
    x, y = (5 * (-1 if flip else 1)), -4
    expected = torch.tensor([15 + scale*(np.cos(radians)*x + np.sin(radians)*y),
                             15 + scale*(-np.sin(radians)*x + np.cos(radians)*y)])
    yy, xx = torch.meshgrid(torch.arange(31), torch.arange(31), indexing='ij')
    centroid = torch.stack(((output[0]*xx).sum(), (output[0]*yy).sum())) / output[0].sum()
    torch.testing.assert_close(centroid.double(), expected, atol=.13, rtol=0)


def test_missing_chandra_and_eval_identity(catalog):
    args = (catalog, ['c0.fits'], ['DE', 'NDE'])
    opts = dict(mode='multimodal', size=16, processed_dir=None)
    ds = ClusterDataset(*args, **opts, augment=True, zoom=zoom((.8, 1.2)))
    for index in range(len(ds)):
        assert ds[index]['chandra'].count_nonzero() == 0
        assert ds[index]['chandra_mask'].count_nonzero() == 0
    evaluation = ClusterDataset(*args, **opts, zoom=zoom((.8, 1.2)))
    assert len(evaluation) == 1 and evaluation.get_item_info(0)['zoom_factor'] == 1
    assert torch.equal(evaluation[0]['raw'], ClusterDataset(*args, **opts)[0]['raw'])


@pytest.mark.parametrize('mode', ['raw', 'pair', 'multimodal'])
def test_scattering_cache_matches_zoomed_pixels(catalog, tmp_path, mode):
    Scattering2D = pytest.importorskip('kymatio.torch').Scattering2D
    options = dict(mode=mode, size=16, J=2, L=2, max_order=2,
                   processed_dir=None, scattering_dir=tmp_path/'scat', verbose=False)
    z = zoom((.8, 1.2))  # deliberately no identity factor
    info = catalog['c2.fits']
    assert prepare_scattering(catalog, ['c2.fits'], [], **options, zoom=z)['created'] == 1
    assert prepare_scattering(catalog, ['c2.fits'], [], **options, zoom=z)['reused'] == 1
    params = {k: options[k] for k in ['mode', 'size', 'J', 'L', 'max_order', 'scattering_dir']}
    coefficients = load_scattering(info, **params, zoom=z)
    assert coefficients.shape[0] == 48
    ds = ClusterDataset(catalog, ['c2.fits'], ['DE','NDE'], size=16, mode=mode,
                        augment=True, zoom=z, processed_dir=None, scattering_dir=options['scattering_dir'],
                        scattering_params={'J': 2, 'L': 2, 'max_order': 2})
    direct = Scattering2D(J=2, L=2, shape=(16,16))
    for i in range(len(ds)):
        sample = ds[i]
        names = ['raw'] if mode == 'raw' else ['raw', 'chandra', 'chandra_mask']
        expected = direct(torch.cat([sample[n] for n in names])[None])[0]
        torch.testing.assert_close(sample['scattering'], expected)
    original_path = scattering_paths(info, **params, zoom=z)['data']
    for changed in [zoom((1.2,.8)), zoom((.9,1.2)), zoom((.8,1.2), padding_mode='reflect'),
                    zoom((.8,1.2), interpolation='nearest'), zoom((.8,1.2), fill_value=.2)]:
        assert scattering_paths(info, **params, zoom=changed)['data'] != original_path
        with pytest.raises(FileNotFoundError):
            load_scattering(info, **params, zoom=changed)
    prepare_scattering(catalog, ['c2.fits'], [], **options)
    assert scattering_paths(info, **params) == scattering_paths(info, **params, zoom={'enabled':False})
    ds.augment = False
    base = ClusterDataset(catalog, ['c2.fits'], ['DE','NDE'], size=16, mode=mode, processed_dir=None)[0]
    names = ['raw'] if mode == 'raw' else ['raw', 'chandra', 'chandra_mask']
    torch.testing.assert_close(ds[0]['scattering'], direct(torch.cat([base[n] for n in names])[None])[0])


@pytest.mark.parametrize('task', ['de_nde', 'rr_rh'])
@pytest.mark.parametrize('model', ['cnn', 'image_cnn', 'fusion', 'dual_encoder_cnn', 'dual_ssn', 'dual_encoder_ssn'])
def test_training_all_architectures_train_only(catalog, tmp_path, task, model):
    if 'ssn' in model:
        pytest.importorskip('kymatio')
    datasets = []
    def create(*args, **kwargs):
        ds = ClusterDataset(*args, **kwargs)
        datasets.append(ds)
        return ds
    splits = (['c2.fits', 'c5.fits'], ['c8.fits','c11.fits'], ['c4.fits','c17.fits'])
    config = ExperimentConfig(name='zoom', size=16, width=2, model=model, mode='pair',
        task=task, device='cpu', epochs=1, batch_size=16, augment=True,
        use_mixup=False, processed_dir=str(tmp_path/'processed'), verbose=False,
        augmentation={'zoom': zoom((.8, 1.2))},
        model_params={'J':2,'L':2,'hidden_dim2':4} if 'ssn' in model else {})
    with patch('clusterprep.training.ClusterDataset', side_effect=create):
        run = run_experiment(catalog, config, tmp_path/'runs', splits=splits)
    assert [len(ds) for ds in datasets] == [96, 2, 2]
    ids = [{ds.get_item_info(i)['cluster_id'] for i in range(len(ds))} for ds in datasets]
    assert not (ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    for ds in datasets[1:]:
        clean = ClusterDataset(catalog, ds.keys, ['DE','NDE'], mode='pair', size=16, processed_dir=None)
        for i in range(len(ds)):
            assert torch.equal(ds[i]['raw'], clean[i]['raw'])
    assert json.loads((run/'metrics.json').read_text())['epochs_ran'] == 1
    summary = json.loads((run/'config.json').read_text())['augmentation_summary']
    assert summary == dict(train_variants_per_cluster=48, train_batches_per_epoch=6,
                           examples_per_split={'train':96, 'validation':2, 'test':2})


def test_visualization_shared_scales(catalog, tmp_path):
    import matplotlib.pyplot as plt
    ds = ClusterDataset(catalog, ['c2.fits'], ['DE','NDE'], mode='pair', size=16, processed_dir=None)
    fig = plot_zoom_variants(ds, output_path=tmp_path/'zooms.png')
    axes = [ax for ax in fig.axes if ax.images]
    assert len(axes) == 12
    for row in range(3):
        assert len({ax.images[0].get_clim() for ax in axes[row*4:(row+1)*4]}) == 1
    assert (tmp_path/'zooms.png').stat().st_size > 0
    plt.close(fig)


@pytest.mark.parametrize('padding', ['constant', 'reflect', 'replicate'])
def test_zoom_composes_with_radio_noise(catalog, padding):
    from clusterprep.augmentation import decode_variant
    options = dict(mode='pair', size=16, augment=True, processed_dir=None,
                   radio_noise=dict(enabled=True, method='gaussian_pixel', probability=1.,
                                    amplitude_range=[.2, .2]), training=True)
    args = (catalog, ['c2.fits'], ['DE', 'NDE'])
    z = zoom((.8,), padding_mode=padding)
    original = ClusterDataset(*args, **{**options, 'augment':False})
    augmented = ClusterDataset(*args, **options, zoom=z)
    names = ['raw','chandra','chandra_mask']
    np.random.seed(123)
    base = original[0]
    np.random.seed(123)
    actual = augmented[7]
    angle, flip, scale = decode_variant(7, z)
    expected = transform_aligned(torch.cat([base[n] for n in names]), angle, flip, scale, z)
    torch.testing.assert_close(torch.cat([actual[n] for n in names]), expected)
    # The normalization bounds come from the original clean image, before zoom.
    np.testing.assert_array_equal(original._noise_cache[0][2], augmented._noise_cache[0][2])
