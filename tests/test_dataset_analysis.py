"""Scientific invariants of the unsplit audit, using tiny synthetic observations."""
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from clusterprep import dataset_analysis as da


def image(path, value=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [4, 4]
    wcs.wcs.cdelt = [-.001, .001]
    wcs.wcs.crval = [180, 45]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    fits.writeto(path, np.full((8, 8), value, dtype=float), wcs.to_header())


def test_inventory_preserves_uncertain_and_missing_objects(tmp_path):
    root = tmp_path / 'data'
    image(root / 'RAW/cDE/A.fits')
    image(root / 'RAW/U/A.fits')
    image(root / 'RAW/NDE/B.fits')
    catalogue = tmp_path / 'catalog.csv'
    catalogue.write_text('slug,z,classification\nA,0.1,cDE\nB,0.2,NDE\nC,0.3,U\n')
    config = dict(data_dir=str(root), catalogue=str(catalogue), legacy={})
    records, clusters, manifest, issues = da.inventory(config)
    assert len(clusters) == 3 and len(manifest) == 3
    assert clusters.radio.sum() == 2 and clusters.xray.sum() == 0
    assert records['A']['labels'] == {'cDE', 'U'}
    targets, _ = da.classification_tables(records)
    assert not targets[targets.cluster_id == 'A'].eligible.any()
    assert not targets[targets.cluster_id == 'C'].eligible.any()
    assert len(issues) == 1  # identical annotation copies, not extra clusters


def test_conflicting_copies_are_not_silently_selected(tmp_path):
    image(tmp_path / 'RAW/DE/A.fits', 1)
    image(tmp_path / 'RAW/RH/A.fits', 2)
    records, clusters, _, issues = da.inventory(dict(data_dir=str(tmp_path), catalogue=str(tmp_path/'missing.csv'), legacy={}))
    assert records['A']['raw_path'] is None
    assert clusters.radio_file_present.all() and not clusters.radio.any()
    assert issues.issue.str.contains('conflicting').any()


def test_candidate_mask_and_class_order():
    records = {'a': dict(raw_path='a.fits', labels={'cRH'}),
               'b': dict(raw_path='b.fits', labels={'NDE'}),
               'c': dict(raw_path='c.fits', labels={'DE'})}
    targets, summary = da.classification_tables(records)
    candidate = targets[(targets.cluster_id == 'a') & (targets.task == 'rr_rh') & (targets.candidate_policy == 'ignore')].iloc[0]
    assert candidate.eligible and candidate.valid_0 and not candidate.valid_1
    nde = targets[(targets.cluster_id == 'b') & (targets.task == 'de_nde')].iloc[0]
    assert nde.target_0 == 1
    assert summary.positives.ge(0).all()


def test_missing_xray_and_cache_do_not_touch_sources(tmp_path):
    source = tmp_path/'RAW/NDE/A.fits'
    image(source)
    before = source.read_bytes()
    config = dict(output_dir=str(tmp_path/'analysis'), size=4, cache=True, alignment=True)
    info = dict(cluster_id='A', raw_path=str(source), chandra_path=None)
    a = da.aligned_sample(info, config)
    b = da.aligned_sample(info, config)
    np.testing.assert_array_equal(a['raw'], b['raw'])
    np.testing.assert_allclose(a['wcs'].pixel_to_world_values(1, 1), b['wcs'].pixel_to_world_values(1, 1))
    assert 'chandra_mask' not in a
    assert source.read_bytes() == before
    table = da.collect_alignment({'A': info}, config)
    assert table.status.eq('ok').all() and table.coverage.isna().all()


def test_pixel_statistics_preserve_missing_and_zero():
    stats = da.pixel_statistics(np.array([[0., 2.], [np.nan, np.inf]]))
    assert stats['finite_fraction'] == .5
    assert stats['zero_fraction'] == .25
    assert stats['median'] == 1
