"""Phase 5: paired RR/RH inventory and provisional splits; never trains models."""
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import torch

from .common_wcs_validation import read_catalog, discover_sources, physical_field
from .dataset import ClusterDataset
from .labels import task_target, labels_for, ANNOTATION_NAMES
from .matched_cache import MatchedPairStore, canonical_json
from .modalities import fit_normalization
from .splits import split_data_keys, multilabel_stratum

NAMES = ('train', 'validation', 'test')
COVERAGES = ('RAW', 'T50kpcSUB', 'CHANDRA', 'intersection')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False))


def write_csv(path, rows):
    names = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def annotation_index(classified_root):
    """ID-based annotations from existing RAW category directories, merged once.

    Product-specific availability never changes annotation membership. Unknown
    categories (U, cDE, unclassified) remain in provenance but give no RR/RH target.
    """
    labels, origins = {}, {}
    root = Path(classified_root)/'RAW'
    if not root.is_dir():
        raise ValueError(f'Annotation directory missing: {root}')
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(directory.glob('*.fits')):
            labels.setdefault(path.stem, set()).add(directory.name)
            origins.setdefault(path.stem, []).append(str(path.resolve()))
    return labels, origins


def cohort_annotation(annotations, candidate_policy='positive'):
    """Preserve positives/candidate policy; absent subtype is UNKNOWN, not zero."""
    labels = set(annotations) & set(ANNOTATION_NAMES)
    labels_for({'labels': labels})  # Reject contradictory NDE/positive annotations.
    known = [int('NDE' in labels or name in labels or
                 (candidate_policy == 'positive' and 'c'+name in labels)) for name in ('RR', 'RH')]
    info = dict(labels=sorted(labels), rr_rh_known_mask=known)
    target, mask = task_target(info, 'rr_rh', candidate_policy)
    if not mask.any():
        raise ValueError('No supervised RR/RH annotation under candidate policy')
    return info, target, mask


def canonical_record(record):
    result = {'cluster_id': record['cluster_id']}
    for name in ('ra_deg', 'dec_deg', 'redshift', 'r500_kpc', 'm500_msun'):
        v = record.get(name)
        result[name] = float(v) if v is not None and np.isfinite(v) else None
    return result


def counts(info, keys, policy='positive'):
    result = dict(n=len(keys), RR_positive=0, RH_positive=0, RR_negative=0, RH_negative=0,
                  RR_unknown=0, RH_unknown=0)
    combos = Counter()
    annotations = Counter()
    for key in keys:
        target, mask = task_target(info[key], 'rr_rh', policy)
        combos[multilabel_stratum(info[key], policy)] += 1
        annotations.update(info[key]['labels'])
        for j, label in enumerate(('RR', 'RH')):
            result[label+('_unknown' if not mask[j] else '_positive' if target[j] else '_negative')] += 1
    result.update({f'combination_{c}': combos[c] for c in ('00', '10', '01', '11', '1?', '?1', '??', '0?', '?0')})
    result.update({f'annotation_{label}': annotations[label] for label in ('RR', 'RH', 'cRR', 'cRH', 'NDE')})
    return result


def coverage_analysis(info, measurements, policy='positive'):
    rows = []
    # 0% is no additional cut on a base that already has strictly positive support.
    for radio in (0., .5, .8, .9, 1.):
        for xray in (0., .1, .2, .3, .5):
            keys = [k for k in info if measurements[k]['coverage']['intersection'] >= radio
                    and measurements[k]['coverage']['CHANDRA'] >= xray]
            rows.append(dict(min_radio_intersection=radio, min_chandra=xray,
                             **counts(info, keys, policy)))
    return rows


def provisional_splits(info, policy='positive', seed=42):
    """New 60/20/20 master assignment. No scientific threshold is selected."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        splits = split_data_keys(info, task='rr_rh', mode='pair', candidate_policy=policy,
                                seed=seed, test_size=.2, val_size=.2)
    assigned = [k for keys in splits for k in keys]
    if len(set(assigned)) != len(assigned) or set(assigned) != set(info):
        raise ValueError('Split overlap or incomplete cohort assignment')
    return dict(schema_version=1, status='provisional_pending_scientific_approval',
                scientific_approval=False, final_coverage_thresholds=None,
                cohort_rule='technical eligibility and strictly positive radio-intersection/X-ray coverage',
                seed=seed, test_fraction=.2, validation_fraction=.2,
                candidate_policy=policy, missing_subtype_policy='unknown_masked',
                preprocessing='matched_resolution', radio_mask='intersection',
                **{n: sorted(keys) for n, keys in zip(NAMES, splits)},
                distributions={n: counts(info, keys, policy) for n, keys in zip(NAMES, splits)},
                stratification='joint label/mask combinations; iterative multilabel fallback for rare strata',
                stratification_warnings=[str(w.message) for w in caught],
                samples=deepcopy(info))


def audit_pretraining(results_root, manifest):
    """Audit every discoverable DE/NDE manifest, not an assumed chosen checkpoint.

    Unverifiable schemas are reported. Exposure includes both supervised training
    and source validation used for checkpoint selection. No automatic transfer.
    """
    target = {n: set(manifest[n]) for n in NAMES}
    reports = []
    for path in sorted(Path(results_root).rglob('splits.json')):
        config_path = path.with_name('config.json')
        try:
            config = json.loads(config_path.read_text())
            config = config.get('config', config)
            task = config.get('task')
            # Never silently infer DE/NDE from an unspecified task or folder name.
            if task != 'de_nde':
                if task is None:
                    reports.append(dict(path=str(path.resolve()), status='task_unknown', reason='config.task absent'))
                continue
            source = json.loads(path.read_text())
            samples = source.get('samples')
            if not isinstance(samples, dict):
                raise ValueError('samples mapping absent; cluster IDs cannot be verified')
            source_ids = {}
            for n in NAMES:
                if not isinstance(source.get(n), list):
                    raise ValueError(f'{n} missing or malformed')
                ids = [samples[k]['cluster_id'] for k in source[n]]
                if len(set(ids)) != len(ids):
                    raise ValueError(f'duplicate cluster in source {n}')
                source_ids[n] = set(ids)
            if any(source_ids[a] & source_ids[b] for a, b in
                   [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
                raise ValueError('source split overlap')
            overlaps = {f'target_{a}_in_source_{b}': sorted(target[a] & source_ids[b])
                        for a, b in [('test', 'train'), ('validation', 'train'), ('test', 'validation')]}
            mismatches = sorted(k for n in NAMES for k in target[n]
                                if any(k in source_ids[m] for m in NAMES if m != n))
            reports.append(dict(path=str(path.resolve()), task=task,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                status='exposure_detected' if any(overlaps.values()) else 'no_detected_exposure',
                checkpoint_exists=path.with_name('best.pt').is_file(),
                source_preprocessing=config.get('radio_preprocessing', 'historical'),
                source_radio=config.get('radio_source', 'RAW'),
                source_counts={n: len(v) for n, v in source_ids.items()},
                partition_mismatches=mismatches, **overlaps))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reports.append(dict(path=str(path.resolve()), status='unverifiable', reason=str(exc)))
    return reports


def make_dataset(info, keys, product, cache_root, policy='positive', normalization=None, bounds=None):
    return ClusterDataset(info, keys, ['RR', 'RH'], mode='multimodal', task='rr_rh',
        candidate_policy=policy, size=128, augment=False, modalities=['radio', 'xray'],
        radio_source=product, radio_preprocessing='matched_resolution',
        matched_resolution={'radio_mask': 'intersection'}, processed_dir=cache_root,
        normalization=normalization, normalization_bounds=bounds)


def fit_train_normalization(info, manifest, cache_root):
    """Fit separately on the manifest's train keys; freeze for validation/test."""
    training = manifest['train']
    if set(training) & (set(manifest['validation']) | set(manifest['test'])):
        raise ValueError('Train/holdout overlap in normalization input')
    options = {m: dict(scope='train', method='asinh_percentile', lower_percentile=30,
                       upper_percentile=99, alpha=10.) for m in ('radio', 'xray')}
    fitted = {}
    for product in ('RAW', 'T50kpcSUB'):
        ds = make_dataset(info, training, product, cache_root, manifest['candidate_policy'], options)
        fitted[product] = fit_normalization(ds)
    if fitted['RAW']['xray'] != fitted['T50kpcSUB']['xray']:
        raise ValueError('X-ray train statistics differ between paired views')
    return dict(status='provisional_master_splits_only', train_keys=list(training),
                train_keys_sha256=hashlib.sha256(canonical_json(sorted(training)).encode()).hexdigest(),
                options=options, products=fitted, frozen_for=['validation', 'test'],
                default_dataset_scope='image', automatically_enabled_for_training=False,
                note='Refit on the approved train subset if scientific coverage selection changes the cohort')


def verify_pairs(info, manifest, cache_root, fitted):
    """Check every member using two actual dataset views, plus linear metadata."""
    rows, errors = [], []
    for split in NAMES:
        keys = manifest[split]
        a = make_dataset(info, keys, 'RAW', cache_root, manifest['candidate_policy'],
                         fitted['options'], fitted['products']['RAW'])
        b = make_dataset(info, keys, 'T50kpcSUB', cache_root, manifest['candidate_policy'],
                         fitted['options'], fitted['products']['T50kpcSUB'])
        for index, key in enumerate(keys):
            try:
                x, y = a[index], b[index]
                ma, mb = a.get_resolution_metadata(index), b.get_resolution_metadata(index)
                checks = dict(cluster_id=x['cluster_id'] == y['cluster_id'] == key,
                    split=a.keys[index] == b.keys[index] == key and key in manifest[split],
                    wcs=ma['output_wcs'] == mb['output_wcs'], beam=ma['target_beam'] == mb['target_beam'],
                    radio_mask=torch.equal(x['raw_mask'], y['raw_mask']),
                    chandra=torch.equal(x['chandra'], y['chandra']),
                    chandra_mask=torch.equal(x['chandra_mask'], y['chandra_mask']),
                    linear_chandra=ma['chandra_sha256'] == mb['chandra_sha256'],
                    linear_chandra_mask=ma['chandra_mask_sha256'] == mb['chandra_mask_sha256'],
                    label=torch.equal(x['label'], y['label']), label_mask=torch.equal(x['label_mask'], y['label_mask']),
                    annotations=a.data_info[key]['labels'] == b.data_info[key]['labels'],
                    finite=all(torch.isfinite(t).all().item() for t in (x['raw'], y['raw'], x['chandra'])))
                failed = [n for n, valid in checks.items() if not valid]
                if failed:
                    errors.append(dict(cluster_id=key, split=split, reason=';'.join(failed)))
                rows.append(dict(cluster_id=key, split=split, status='error' if failed else 'ok',
                                 **{'check_'+n: valid for n, valid in checks.items()}))
            except Exception as exc:
                errors.append(dict(cluster_id=key, split=split, reason=f'{type(exc).__name__}: {exc}'))
                rows.append(dict(cluster_id=key, split=split, status='error'))
    return rows, errors


def build_cohort(project, output, *, candidate_policy='positive', seed=42):
    """Export a new provisional cohort. Refuses existing output directories."""
    project, output = Path(project).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog_path = project/'outputs/common_wcs_validation/inputs/lofar_psz2_catalog.fits'
    catalog = {r['cluster_id']: canonical_record(r) for r in read_catalog(catalog_path)}
    annotations, origins = annotation_index(project/'scratch/data/PSZ2/classified')
    keys = sorted(set(catalog) | set(annotations))
    cache_root = output/'cache'
    store = MatchedPairStore(cache_root)
    parameters = dict(pixels_per_beam=3., min_support=.999, truncate=5., output_unit='Jy/beam')
    rows, retained, measurements = [], {}, {}
    for i, cid in enumerate(keys):
        record = catalog.get(cid, {'cluster_id': cid})
        sources = discover_sources(record, project/'scratch/data/PSZ2/fits')
        row = dict(cluster_id=cid, annotations=';'.join(sorted(annotations.get(cid, []))),
                   labels='', RR='', RH='', RR_known='', RH_known='', combination='',
                   has_RAW=Path(sources['RAW']).is_file(), has_T50kpcSUB=Path(sources['T50kpcSUB']).is_file(),
                   has_CHANDRA=Path(sources['CHANDRA']).is_file(), has_physical_metadata=physical_field(record) is not None,
                   annotations_usable=False, status='excluded', reason='', fits_validated=False,
                   coverage_RAW='', coverage_T50kpcSUB='', coverage_CHANDRA='', coverage_intersection='',
                   fwhm_arcsec='', fwhm_kpc='', fwhm_over_r500='', cache_key='', split='')
        reasons = []
        try:
            label_info, target, mask = cohort_annotation(annotations.get(cid, []), candidate_policy)
            row.update(labels=';'.join(label_info['labels']), RR=int(target[0]) if mask[0] else '?',
                       RH=int(target[1]) if mask[1] else '?', RR_known=int(mask[0]), RH_known=int(mask[1]),
                       combination=multilabel_stratum(label_info, candidate_policy), annotations_usable=True)
        except ValueError as exc:
            reasons.append(f'annotations: {exc}')
        for name in ('RAW', 'T50kpcSUB', 'CHANDRA'):
            if not row['has_'+name]:
                reasons.append(f'missing_{name}')
        if not row['has_physical_metadata']:
            reasons.append('missing_physical_metadata')
        if not reasons:
            try:
                pair, _ = store.get({n: sources[n] for n in ('RAW', 'T50kpcSUB', 'CHANDRA')}, record, 128, parameters)
                metadata = pair.metadata
                row.update(fits_validated=True, cache_key=metadata['cache_key'],
                           **{'coverage_'+n: metadata['coverage'][n] for n in COVERAGES},
                           **{n: metadata['target_resolution'][n] for n in ('fwhm_arcsec', 'fwhm_kpc', 'fwhm_over_r500')})
                if metadata['coverage']['intersection'] <= 0 or metadata['coverage']['CHANDRA'] <= 0:
                    reasons.append('no_common_radio_or_chandra_support')
                else:
                    retained[cid] = dict(cluster_id=cid, raw_path=sources['RAW'],
                        radio_paths={n: sources[n] for n in ('RAW', 'T50kpcSUB')},
                        chandra_path=sources['CHANDRA'], physical_record=record,
                        annotation_paths=origins.get(cid, []), **label_info)
                    measurements[cid] = metadata
                    row['status'] = 'technically_eligible'
            except Exception as exc:
                reasons.append(f'preprocessing: {type(exc).__name__}: {exc}')
        row['reason'] = '; '.join(reasons)
        rows.append(row)
        # Flush diagnostics after every member; the physical cache remains reusable.
        write_csv(output/'matched_cohort.csv', rows)
        if i % 10 == 0 or i+1 == len(keys):
            print(f'{i+1}/{len(keys)} inspected; {len(retained)} technically eligible', flush=True)
    manifest = provisional_splits(retained, candidate_policy, seed)
    manifest.update(cohort_counts=counts(retained, list(retained), candidate_policy),
                    catalogue_sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
                    cohort_cache_keys={k: measurements[k]['cache_key'] for k in retained},
                    cache_root=str(cache_root))
    # Repeat to verify deterministic membership, not merely a stored seed.
    repeated = provisional_splits(retained, candidate_policy, seed)
    if any(manifest[n] != repeated[n] for n in NAMES):
        raise RuntimeError('Non-reproducible splits')
    membership = {k: n for n in NAMES for k in manifest[n]}
    for row in rows:
        row['split'] = membership.get(row['cluster_id'], '')
    write_csv(output/'matched_cohort.csv', rows)
    write_json(output/'matched_splits.json', manifest)
    analysis = coverage_analysis(retained, measurements, candidate_policy)
    write_csv(output/'coverage_analysis.csv', analysis)
    audits = audit_pretraining(project/'results', manifest)
    write_json(output/'pretraining_audit.json', audits)
    fitted = fit_train_normalization(retained, manifest, cache_root)
    write_json(output/'normalization_audit.json', fitted)
    checked, errors = verify_pairs(retained, manifest, cache_root, fitted)
    write_csv(output/'pair_validation.csv', checked)
    write_json(output/'pair_errors.json', errors)
    from .matched_cohort_report import write_report
    write_report(output, rows, manifest, measurements, analysis, audits, fitted, errors)
    print(json.dumps(dict(inventory=len(rows), retained=len(retained), errors=len(errors),
                          counts=manifest['cohort_counts']), ensure_ascii=False), flush=True)
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--candidate-policy', choices=['positive', 'ignore'], default='positive')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    build_cohort(args.project, args.output or args.project/'outputs/matched_cohort_phase5',
                 candidate_policy=args.candidate_policy, seed=args.seed)


if __name__ == '__main__':
    main()
