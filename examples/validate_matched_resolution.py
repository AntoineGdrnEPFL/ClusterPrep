"""Small phase 1–3 demonstration; no dataset integration or training.

Run from the project root:
  .venv/bin/python clusterprep/examples/validate_matched_resolution.py
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import AsinhNorm
from matplotlib.patches import Ellipse

from clusterprep.common_wcs_validation import read_catalog, discover_sources, fingerprint
from clusterprep.matched_resolution import prepare_matched_pair
from clusterprep.radio_psf import read_radio, jacobian, Beam, FWHM_SIGMA


def figure(pair, sources, path):
    fig = Figure(figsize=(14, 9), layout='constrained'); FigureCanvasAgg(fig)
    native = [read_radio(sources[n]) for n in ('RAW', 'T50kpcSUB')]
    panels = [(native[0].data, native[0].wcs, 'RAW natif', native[0].bunit, native[0].beam),
              (native[1].data, native[1].wcs, 'T50kpcSUB natif', native[1].bunit, native[1].beam),
              (pair.chandra, pair.wcs, 'CHANDRA partagé', pair.metadata.get('chandra_bunit'), None),
              (pair.raw, pair.wcs, 'RAW harmonisé', 'Jy/beam cible', Beam(**pair.metadata['target_beam'])),
              (pair.t50kcpsub, pair.wcs, 'T50kpcSUB harmonisé', 'Jy/beam cible', Beam(**pair.metadata['target_beam'])),
              (np.where(pair.common_radio_mask, pair.raw-pair.t50kcpsub, np.nan), pair.wcs,
               'RAW − T50kpcSUB', 'Jy/beam cible', None)]
    for i, (data, wcs, title, unit, beam) in enumerate(panels):
        ax = fig.add_subplot(2, 3, i+1, projection=wcs)
        finite = data[np.isfinite(data)]
        lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0., 1.)
        hi = max(hi, lo+1e-12)
        norm = AsinhNorm(linear_width=(hi-lo)/10, vmin=lo, vmax=hi)
        artist = ax.imshow(np.ma.masked_invalid(data), origin='lower', cmap='inferno', norm=norm)
        j = jacobian(wcs, (data.shape[1]-1)/2, (data.shape[0]-1)/2)
        ax.set_title(title + f'\n{np.linalg.norm(j, axis=0).round(2)} arcsec/pix', fontsize=10)
        ax.set_xlabel('RA'); ax.set_ylabel('Dec')
        if beam:
            cov = np.linalg.inv(j) @ beam.covariance @ np.linalg.inv(j).T
            values, vec = np.linalg.eigh(cov)
            widths = FWHM_SIGMA*np.sqrt(values)
            ax.add_patch(Ellipse((.12*data.shape[1], .12*data.shape[0]), widths[1], widths[0],
                                 angle=np.degrees(np.arctan2(vec[1, 1], vec[0, 1])),
                                 edgecolor='cyan', facecolor='none', lw=1.5))
        bar = fig.colorbar(artist, ax=ax, shrink=.65, label=unit or 'unité FITS non renseignée',
                          ticks=norm.inverse(np.linspace(0, 1, 5)), format='%.2g')
        bar.minorticks_off()
    meta = pair.metadata
    fig.suptitle(f"{meta['cluster_id']} — côté 4R500 : {meta['field']['fov_arcmin']:.2f} arcmin\n"
                 f"Beam cible : {meta['target_beam']['major_arcsec']:.2f} arcsec ; "
                 "rendus P1–P99 indépendants ; ellipse cyan : beam nominal")
    fig.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--clusters', nargs='+', default=['PSZ2G023.17+86.71', 'PSZ2G033.81+77.18', 'PSZ2G048.75+53.18'])
    args = parser.parse_args()
    root = args.project.resolve()
    out = (args.output or root/'outputs/matched_resolution_phase123').resolve()
    # A demo never overwrites a previous run or any training output.
    out.mkdir(parents=True, exist_ok=False)
    records = {r['cluster_id']: r for r in read_catalog(root/'outputs/common_wcs_validation/inputs/lofar_psz2_catalog.fits')}
    rows = []
    for cid in args.clusters:
        row = dict(cluster_id=cid, status='error', reason='', beam_arcsec=None,
                   raw_coverage=None, t50_coverage=None, chandra_coverage=None)
        try:
            record = records[cid]
            sources = discover_sources(record, root/'scratch/data/PSZ2/fits')
            pair = prepare_matched_pair(sources['RAW'], sources['T50kpcSUB'], sources['CHANDRA'], record)
            directory = out/cid; directory.mkdir()
            for name in ('raw', 't50kcpsub', 'chandra'):
                header = pair.wcs.to_header()
                if name != 'chandra':
                    width = pair.metadata['target_beam']['major_arcsec']/3600
                    header.update(BMAJ=width, BMIN=width, BPA=0., BUNIT='Jy/beam')
                elif pair.metadata['chandra_bunit']:
                    header['BUNIT'] = pair.metadata['chandra_bunit']
                fits.writeto(directory/f'{name}.fits', getattr(pair, name), header, checksum=True)
                mask_header = pair.wcs.to_header(); mask_header['BUNIT'] = '1'
                fits.writeto(directory/f'{name}_mask.fits', getattr(pair, name+'_mask').astype('uint8'), mask_header, checksum=True)
            np.savez_compressed(directory/'tensors.npz', **{k:v.numpy() for k,v in pair.tensors().items()})
            pair.metadata['source_fingerprints'] = {n: fingerprint(sources[n]) for n in ('RAW', 'T50kpcSUB', 'CHANDRA')}
            (directory/'metadata.json').write_text(json.dumps(pair.metadata, indent=2, allow_nan=False))
            figure(pair, sources, directory/'comparison.png')
            row.update(status='ok', beam_arcsec=pair.metadata['target_beam']['major_arcsec'],
                       raw_coverage=float(pair.raw_mask.mean()), t50_coverage=float(pair.t50kcpsub_mask.mean()),
                       chandra_coverage=float(pair.chandra_mask.mean()))
        except Exception as exc:
            row['reason'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)
        print(row, flush=True)
    with (out/'resolution_metadata.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'rejected.json').write_text(json.dumps([r for r in rows if r['status'] != 'ok'], indent=2))


if __name__ == '__main__':
    main()
