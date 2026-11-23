"""Geometry, photometry separation, missing data, and resumable export contracts."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from astropy.io import fits
from astropy import units as u
from astropy.table import Table
from astropy.cosmology import Planck18
from astropy.wcs import WCS
from clusterprep.common_wcs_validation import (
    ValidationConfig, common_grid, median_field, physical_field, preprocess_common,
    project_plane, run_validation, read_catalog, NORMALIZATION, fingerprint,
)
from clusterprep.preprocessing import normalize

RECORD = dict(cluster_id='test', ra_deg=30., dec_deg=20., redshift=.2,
              r500_kpc=1000., m500_msun=5e14)


def native_header(kind='CDELT'):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [30, 20]
    w.wcs.crpix = [32.5, 32.5]
    w.wcs.cdelt = [-.01, .01]
    h = w.to_header()
    angle = np.deg2rad(17)
    pc = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    if kind == 'PC':
        for i in range(2):
            for j in range(2):
                h[f'PC{i+1}_{j+1}'] = pc[i, j]
    if kind == 'CD':
        del h['CDELT1']; del h['CDELT2']
        matrix = np.diag([-.01, .01]) @ pc
        for i in range(2):
            for j in range(2):
                h[f'CD{i+1}_{j+1}'] = matrix[i, j]
    h['BUNIT'] = 'Jy/beam'
    h['BMAJ'] = .02; h['BMIN'] = .01; h['BPA'] = 17.
    return h


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / 'sources'
    directory = root / 'test'; directory.mkdir(parents=True)
    data = np.full((64, 64), -2., dtype=np.float32)
    data[25:35, 25:35] = 0  # physical zero must stay valid
    for suffix in ('', 'T50kpcSUB', 'CHANDRA'):
        fits.PrimaryHDU(data, native_header()).writeto(directory / f'test{suffix}.fits')
    h = native_header(); h['BAND0'] = 'g'; h['BAND1'] = 'z'
    fits.PrimaryHDU(np.stack([data, data + 1]), h).writeto(directory / 'test_legacy.fits')
    return root


def sources(root):
    return {n: str(root / 'test' / f'test{s}.fits') for n, s in
            [('RAW', ''), ('T50kpcSUB', 'T50kpcSUB'), ('CHANDRA', 'CHANDRA'), ('LEGACY', '_legacy')]}


@pytest.mark.parametrize('kind', ['CDELT', 'PC', 'CD'])
def test_celestial_positions_and_masks(kind):
    source = WCS(native_header(kind)).celestial
    y, x = np.mgrid[:64, :64]
    # A linear sky-coordinate field tests reprojected coordinates, not just headers.
    ra, dec = source.pixel_to_world_values(x, y)
    data = ra + 2 * dec
    target, _ = common_grid(RECORD, 20.)
    projected, mask = project_plane(data, source, target, 128)
    ty, tx = np.mgrid[:128, :128]
    tra, tdec = target.pixel_to_world_values(tx, ty)
    np.testing.assert_allclose(projected[mask], (tra + 2*tdec)[mask], atol=1e-5)
    assert mask.any()
    # Larger field has true missing areas, not invented zeros.
    large, _ = common_grid({**RECORD, 'redshift': None}, 80.)
    projected, mask = project_plane(np.zeros((64, 64)), source, large, 128)
    assert mask.any() and not mask.all()
    assert np.isnan(projected[~mask]).all()
    assert (projected[mask] == 0).all()


def test_shared_grid_and_radio_independence(inputs):
    target, _ = common_grid(RECORD, 20.)
    images, errors, missing = preprocess_common(sources(inputs), target)
    assert not errors and 'LEGACY_r' in missing
    assert set(images) == {'RAW', 'T50kpcSUB', 'CHANDRA', 'LEGACY_g', 'LEGACY_z'}
    for image in images.values():
        assert image['data'].shape == (128, 128)
        assert image['wcs'].to_header() == target.to_header()
        assert np.min(image['data'][image['mask']]) < 0
    no_radio = {k: v for k, v in sources(inputs).items() if k not in ('RAW', 'T50kpcSUB')}
    second, _, _ = preprocess_common(no_radio, target)
    for key in second:
        np.testing.assert_array_equal(images[key]['data'], second[key]['data'])
        np.testing.assert_array_equal(images[key]['mask'], second[key]['mask'])
    np.testing.assert_array_equal(images['RAW']['data'], images['T50kpcSUB']['data'])
    # Same target sky corners and centre for every modality.
    for item in images.values():
        np.testing.assert_allclose(item['wcs'].pixel_to_world_values([0, 63.5, 127], [0, 63.5, 127]),
                                   target.pixel_to_world_values([0, 63.5, 127], [0, 63.5, 127]))


def test_physical_fields_and_fallback():
    derived = physical_field({**RECORD, 'r500_kpc': None})
    rho = Planck18.critical_density(.2).to_value(u.Msun/u.kpc**3)
    np.testing.assert_allclose(4*np.pi/3 * 500 * rho * derived['r500_kpc']**3, 5e14)
    assert derived['fov_method'] == 'm500_derived'
    fallback, n = median_field([RECORD, {**RECORD, 'redshift': None}])
    assert n == 1
    target, f = common_grid({**RECORD, 'redshift': None}, fallback)
    assert f['fov_method'] == 'angular_fallback' and f['fov_arcmin'] == fallback
    np.testing.assert_allclose(target.pixel_to_world_values(63.5, 63.5), [30, 20], atol=1e-10)
    with pytest.raises(ValueError, match='No valid'):
        median_field([{**RECORD, 'redshift': -1}])
    with pytest.raises(ValueError, match='RA/Dec'):
        common_grid({**RECORD, 'ra_deg': None}, fallback)


def test_catalog_units(tmp_path):
    table = Table({'Name': ['test'], 'RA': [30.]*u.deg, 'DEC': [20.]*u.deg,
                   'z': [.2], 'R500': [1.]*u.Mpc, 'M500': [5.]*u.Unit('1e14 solMass')})
    path = tmp_path / 'catalog.fits'; table.write(path)
    record = read_catalog(path)[0]
    assert record['cluster_id'] == RECORD['cluster_id']
    for key in RECORD.keys() - {'cluster_id'}:
        assert record[key] == pytest.approx(RECORD[key])
    table['R500'].unit = None
    table.write(path, overwrite=True)
    with pytest.raises(ValueError, match='units'):
        read_catalog(path)


def test_exports_resume_invalidation_and_isolation(inputs, tmp_path):
    config = ValidationConfig(save_figures=False)
    output = tmp_path / 'validation'
    historical = tmp_path / 'historical_cache'; historical.mkdir()
    sentinel = historical / 'existing.fits'; sentinel.write_bytes(b'unchanged training data')
    before = {str(p): fingerprint(p) for p in inputs.rglob('*.fits')}
    first = run_validation([RECORD], inputs, output, config=config, progress=False)
    assert first[0]['status'] == 'ok'
    directory = output / 'processed/test'
    meta = json.loads((directory / 'metadata.json').read_text())
    target = WCS(fits.Header(meta['wcs']))
    for p in directory.glob('*.fits'):
        with fits.open(p, checksum=True) as hd:
            assert hd[0].verify_checksum() == 1
            assert hd[0].data.shape == (128, 128)
            np.testing.assert_allclose(WCS(hd[0].header).pixel_to_world_values([0, 127], [0, 127]),
                                       target.pixel_to_world_values([0, 127], [0, 127]))
    image = fits.getdata(directory / 'RAW.fits')
    mask = fits.getdata(directory / 'RAW_MASK.fits').astype(bool)
    from clusterprep import common_wcs_validation as module
    assert module.normalize is normalize
    np.testing.assert_array_equal(module.normalize(image, mask=mask, **NORMALIZATION), normalize(image, mask))
    time = (directory / 'RAW.fits').stat().st_mtime_ns
    assert run_validation([RECORD], inputs, output, config=config, progress=False)[0]['resumed']
    assert time == (directory / 'RAW.fits').stat().st_mtime_ns
    # Output corruption must invalidate reuse.
    (directory / 'RAW.fits').write_bytes(b'corrupt')
    assert not run_validation([RECORD], inputs, output, config=config, progress=False)[0]['resumed']
    assert before == {str(p): fingerprint(p) for p in inputs.rglob('*.fits')}
    assert sentinel.read_bytes() == b'unchanged training data'
    # Source change and configuration change also invalidate.
    p = inputs / 'test/test.fits'
    with fits.open(p, mode='update') as hd:
        hd[0].data += 1
    assert not run_validation([RECORD], inputs, output, config=config, progress=False)[0]['resumed']
    assert not run_validation([RECORD], inputs, output, config=replace(config, fov_r500_factor=3), progress=False)[0]['resumed']
    with pytest.raises(ValueError, match='independent'):
        run_validation([RECORD], inputs, inputs / 'output', progress=False)


def test_errors_are_isolated(inputs, tmp_path):
    (inputs / 'test/testT50kpcSUB.fits').write_bytes(b'broken')
    rows = run_validation([RECORD, {**RECORD, 'cluster_id': 'bad', 'dec_deg': 100}],
                          inputs, tmp_path / 'validation', config=ValidationConfig(save_figures=False), progress=False)
    assert [r['status'] for r in rows] == ['partial', 'error']
    assert (tmp_path / 'validation/processed/test/CHANDRA.fits').exists()
    assert 'T50kpcSUB' in rows[0]['error_message']
    assert 'RA/Dec' in rows[1]['error_message']


def test_different_native_radio_grids_do_not_move_other_modalities(inputs):
    target, _ = common_grid(RECORD, 20.)
    baseline, _, _ = preprocess_common(sources(inputs), target)
    # Replace only the radio reference with a different size, rotation and CD WCS.
    h = native_header('CD')
    h['CRPIX1'] = 48.5; h['CRPIX2'] = 48.5
    fits.PrimaryHDU(np.full((96, 96), -2., dtype=np.float32), h).writeto(
        inputs / 'test/test.fits', overwrite=True)
    changed, errors, _ = preprocess_common(sources(inputs), target)
    assert not errors
    assert changed['RAW']['wcs'].to_header() == changed['T50kpcSUB']['wcs'].to_header()
    for key in ('CHANDRA', 'LEGACY_g', 'LEGACY_z'):
        np.testing.assert_array_equal(baseline[key]['data'], changed[key]['data'])
        np.testing.assert_array_equal(baseline[key]['mask'], changed[key]['mask'])


@pytest.mark.parametrize('view', ['before', 'after', 'both'])
def test_figures_save_close_and_reuse_normalization(inputs, tmp_path, monkeypatch, view):
    import matplotlib.pyplot as plt
    import clusterprep.common_wcs_validation as module
    target, field = common_grid(RECORD, 20.)
    images, errors, missing = preprocess_common(sources(inputs), target)
    calls = []
    def checked_normalize(image, mask=None, **kwargs):
        result = normalize(image, mask=mask, **kwargs)
        np.testing.assert_array_equal(result, normalize(image, mask=mask))
        calls.append(kwargs)
        return result
    monkeypatch.setattr(module, 'normalize', checked_normalize)
    def never_show(*args, **kwargs):
        raise AssertionError('Figures must not be displayed')
    monkeypatch.setattr(plt, 'show', never_show)
    previous = plt.get_fignums()
    path = tmp_path / f'{view}.png'
    module.save_figure(path, RECORD, field, images, missing, errors,
                       ValidationConfig(view=view, dpi=40))
    assert path.read_bytes().startswith(b'\x89PNG')
    assert plt.get_fignums() == previous
    assert len(calls) == (0 if view == 'before' else len(images))
    assert all(c == NORMALIZATION for c in calls)
