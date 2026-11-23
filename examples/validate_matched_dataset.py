"""Validate dataset/cache integration on local FITS, without splits or training.

Run from the project root:
  .venv/bin/python clusterprep/examples/validate_matched_dataset.py
"""
import argparse
import json
from pathlib import Path

import torch
from clusterprep import ClusterDataset
from clusterprep.common_wcs_validation import read_catalog, discover_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.project.resolve()
    output = args.output or root/'outputs/matched_resolution_phase4'
    output.mkdir(parents=True, exist_ok=False)
    records = {r['cluster_id']: r for r in read_catalog(root/'outputs/common_wcs_validation/inputs/lofar_psz2_catalog.fits')}
    ids = ['PSZ2G023.17+86.71', 'PSZ2G033.81+77.18', 'PSZ2G048.75+53.18']
    info = {}
    for cid in ids:
        sources = discover_sources(records[cid], root/'scratch/data/PSZ2/fits')
        info[cid] = dict(cluster_id=cid, raw_path=sources['RAW'],
                         radio_paths={n: sources[n] for n in ('RAW', 'T50kpcSUB')},
                         chandra_path=sources['CHANDRA'])
    # Thresholds are demonstration values, not scientifically approved exclusions.
    options = dict(radio_mask='intersection', coverage_policy='report',
                   coverage_thresholds={'RAW': .8, 'T50kpcSUB': .8, 'CHANDRA': .2, 'intersection': .8})
    kwargs = dict(mode='pair', size=128, include_targets=False,
                  radio_preprocessing='matched_resolution', matched_resolution=options,
                  matched_records=records, processed_dir=output/'cache', cache=True)
    raw = ClusterDataset(info, ids, [], radio_source='RAW', **kwargs)
    tapered = ClusterDataset(info, ids, [], radio_source='T50kpcSUB', **kwargs)
    rows = []
    for index, cid in enumerate(ids):
        a, b = raw[index], tapered[index]
        for name in ('chandra', 'chandra_mask', 'raw_mask'):
            assert torch.equal(a[name], b[name]), f'{cid}: {name} differs'
        ma, mb = raw.get_resolution_metadata(index), tapered.get_resolution_metadata(index)
        assert ma['cache_key'] == mb['cache_key']
        assert ma['chandra_sha256'] == mb['chandra_sha256']
        assert ma['chandra_mask_sha256'] == mb['chandra_mask_sha256']
        assert raw.keys == tapered.keys == ids
        (output/f'{cid}.json').write_text(json.dumps({'RAW': ma, 'T50kpcSUB': mb}, indent=2, allow_nan=False))
        row = dict(cluster_id=cid, coverage=ma['coverage'], target_resolution=ma['target_resolution'],
                   eligibility=ma['eligibility'], same_chandra=True, same_chandra_mask=True,
                   same_radio_mask=True, shared_pair_cache=True)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (output/'verification.json').write_text(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
