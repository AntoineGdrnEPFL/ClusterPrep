"""Cohort/split/normalization invariants. No models or training invoked."""
from copy import deepcopy
import json

import numpy as np
import pytest

from clusterprep.labels import task_target
from clusterprep.matched_cohort import (cohort_annotation, counts, coverage_analysis,
    provisional_splits, audit_pretraining, verify_pairs, fit_train_normalization)


def annotated(name, labels, policy='positive'):
    item, _, _ = cohort_annotation(labels, policy)
    return dict(cluster_id=name, chandra_path='present', **item)


@pytest.mark.parametrize('labels,target,mask', [
    (['RH'], [0, 1], [0, 1]), (['RR'], [1, 0], [1, 0]),
    (['NDE'], [0, 0], [1, 1]), (['RR', 'RH'], [1, 1], [1, 1]),
    (['cRR', 'cRH'], [1, 1], [1, 1])])
def test_missing_subtypes_are_unknown_and_candidates_preserved(labels, target, mask):
    item, actual, known = cohort_annotation(labels)
    np.testing.assert_array_equal(actual, target)
    np.testing.assert_array_equal(known, mask)
    np.testing.assert_array_equal(task_target(item, 'rr_rh')[1], mask)


def test_candidate_ignore_and_historical_masks():
    with pytest.raises(ValueError, match='No supervised'):
        cohort_annotation(['cRR'], 'ignore')
    _, target, mask = cohort_annotation(['RH', 'cRR'], 'ignore')
    np.testing.assert_array_equal(mask, [0, 1])
    np.testing.assert_array_equal(target, [0, 1])
    np.testing.assert_array_equal(task_target({'labels': ['RH']}, 'rr_rh')[1], [1, 1])
    for labels in ([], ['DE'], ['U'], ['NDE', 'RH']):
        with pytest.raises(ValueError):
            cohort_annotation(labels)
    with pytest.raises(ValueError, match='binary'):
        task_target({'labels': ['RH'], 'rr_rh_known_mask': [1, .5]}, 'rr_rh')


@pytest.fixture
def population():
    return {f'C{i:03}': annotated(f'C{i:03}', labels) for i, labels in enumerate(
        [['NDE'], ['RR'], ['RH'], ['RR', 'RH']]*15)}


def test_master_splits_disjoint_reproducible_preserve_joint_states(population):
    a = provisional_splits(population)
    b = provisional_splits(dict(reversed(list(population.items()))))
    assert a['final_coverage_thresholds'] is None and not a['scientific_approval']
    assert [len(a[n]) for n in ('train', 'validation', 'test')] == [36, 12, 12]
    assert set(a['train']).isdisjoint(a['test'])
    assert set(a['validation']).isdisjoint(a['train']+a['test'])
    assert set(a['train']+a['validation']+a['test']) == set(population)
    for n in ('train', 'validation', 'test'):
        assert a[n] == b[n]
        assert all(a['distributions'][n]['combination_'+s] > 0 for s in ('00', '1?', '?1', '11'))


def test_threshold_counts_include_unknowns_and_exact_boundaries():
    info = {k: annotated(k, labels) for k, labels in
            [('a', ['NDE']), ('b', ['cRR']), ('c', ['RH']), ('d', ['RR', 'RH'])]}
    measurements = {k: {'coverage': {'intersection': radio, 'CHANDRA': xray}} for k, radio, xray in
                    [('a', 1., .1), ('b', .8, .2), ('c', .5, .3), ('d', 1., .5)]}
    rows = coverage_analysis(info, measurements)
    assert len(rows) == 25
    row = next(r for r in rows if r['min_chandra'] == .2 and r['min_radio_intersection'] == .8)
    assert row['n'] == 2 and row['RR_positive'] == 2 and row['RH_positive'] == 1
    assert row['combination_1?'] == row['combination_11'] == 1
    assert row['RH_unknown'] == 1 and row['combination_10'] == 0


def test_pretraining_audit_reports_all_exposure_types_and_unverifiable(tmp_path, population):
    master = provisional_splits(population)
    root = tmp_path/'source'; root.mkdir()
    source = {'train': [master['test'][0], master['validation'][0]],
              'validation': [master['test'][1]], 'test': []}
    source['samples'] = {k: {'cluster_id': k} for n in ('train', 'validation') for k in source[n]}
    (root/'config.json').write_text(json.dumps({'config': {'task': 'de_nde'}}))
    (root/'splits.json').write_text(json.dumps(source))
    bad = tmp_path/'bad'; bad.mkdir()
    (bad/'config.json').write_text(json.dumps({'task': 'de_nde'}))
    (bad/'splits.json').write_text('{}')
    before = (root/'splits.json').read_bytes()
    reports = audit_pretraining(tmp_path, master)
    report = next(r for r in reports if r['status'] == 'exposure_detected')
    assert report['target_test_in_source_train'] == [master['test'][0]]
    assert report['target_validation_in_source_train'] == [master['validation'][0]]
    assert report['target_test_in_source_validation'] == [master['test'][1]]
    assert any(r['status'] == 'unverifiable' for r in reports)
    assert (root/'splits.json').read_bytes() == before


def test_global_normalization_only_uses_train_and_freezes_holdouts(monkeypatch, population):
    manifest = provisional_splits(population)
    calls = []
    def fake_dataset(info, keys, product, cache_root, policy, options):
        calls.append((product, list(keys)))
        return product
    monkeypatch.setattr('clusterprep.matched_cohort.make_dataset', fake_dataset)
    monkeypatch.setattr('clusterprep.matched_cohort.fit_normalization',
                        lambda product: {'radio': [[0., 1. if product == 'RAW' else 2.]], 'xray': [[0., 3.]]})
    result = fit_train_normalization(population, manifest, 'cache')
    assert calls == [('RAW', manifest['train']), ('T50kpcSUB', manifest['train'])]
    assert result['products']['RAW']['radio'] != result['products']['T50kpcSUB']['radio']
    assert result['products']['RAW']['xray'] == result['products']['T50kpcSUB']['xray']
    bad = deepcopy(manifest); bad['test'].append(manifest['train'][0])
    with pytest.raises(ValueError, match='overlap'):
        fit_train_normalization(population, bad, 'cache')


@pytest.mark.parametrize('mismatch', [False, True])
def test_pair_validator_catches_mismatches(monkeypatch, mismatch):
    import torch
    class FakeDataset:
        def __init__(self, product, keys): self.product, self.keys, self.data_info = product, keys, {'a': {'labels': ['RH']}}
        def __getitem__(self, index):
            zero = torch.zeros(1, 2, 2)
            return {'cluster_id': 'a', 'raw': zero, 'raw_mask': zero,
                    'chandra': zero if self.product == 'RAW' or not mismatch else zero+1,
                    'chandra_mask': zero, 'label': torch.tensor([0., 1.]), 'label_mask': torch.tensor([0., 1.])}
        def get_resolution_metadata(self, index):
            return dict(output_wcs={}, target_beam={}, chandra_sha256=self.product if mismatch else 'same', chandra_mask_sha256='same')
    monkeypatch.setattr('clusterprep.matched_cohort.make_dataset', lambda info, keys, product, *a: FakeDataset(product, keys))
    manifest = dict(train=['a'], validation=[], test=[], candidate_policy='positive')
    fit = dict(options={}, products={'RAW': {}, 'T50kpcSUB': {}})
    rows, errors = verify_pairs({'a': {}}, manifest, 'cache', fit)
    assert len(rows) == 1
    assert rows[0]['cluster_id'] == 'a' and rows[0]['split'] == 'train'
    assert rows[0]['check_cluster_id'] and rows[0]['check_split']
    if mismatch:
        assert rows[0]['status'] == 'error' and len(errors) == 1
        assert errors[0]['reason'] == 'chandra;linear_chandra'
    else:
        assert rows[0]['status'] == 'ok' and errors == []
        assert all(v for k, v in rows[0].items() if k.startswith('check_'))


def test_real_dataset_unknown_mask_propagates(tmp_path):
    from astropy.io import fits
    from test_matched_resolution import grid
    from clusterprep.matched_cohort import make_dataset
    w = grid()
    header = w.to_header(); header.update(BMAJ=9/3600, BMIN=5/3600, BPA=30., BUNIT='Jy/beam')
    path = tmp_path/'image.fits'
    fits.writeto(path, np.ones((256, 256), dtype='float32'), header)
    info = {'a': dict(cluster_id='a', raw_path=str(path), radio_paths={'T50kpcSUB': str(path)},
                     chandra_path=str(path), physical_record=dict(ra_deg=170., dec_deg=30., redshift=.2, r500_kpc=100.),
                     **cohort_annotation(['RH'])[0])}
    for product in ('RAW', 'T50kpcSUB'):
        sample = make_dataset(info, ['a'], product, tmp_path/'cache')[0]
        np.testing.assert_array_equal(sample['label'].numpy(), [0, 1])
        np.testing.assert_array_equal(sample['label_mask'].numpy(), [0, 1])
