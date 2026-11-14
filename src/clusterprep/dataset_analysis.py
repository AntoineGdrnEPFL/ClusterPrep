"""Read-only, pre-split observational audit; writes only to its analysis directory.

The tolerant inventory deliberately includes categories excluded by the training
inventory. Targets, FITS readers, preprocessing and augmentation use project code.
"""
from pathlib import Path
import hashlib
import inspect
import json
import platform
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales
from tqdm.auto import tqdm

from . import io, preprocessing
from .labels import ANNOTATION_NAMES, task_target
from .augmentation import variant_count, transform_aligned

MODALITIES = ('radio', 'xray', 'optical')


def read_config(path, project_root):
    root = Path(project_root).resolve()
    config = json.loads(Path(path).read_text())
    training_path = root / config['training_config']
    training = json.loads(training_path.read_text())
    experiment = training['experiments'][0]
    if experiment.get('radio_source', 'RAW') != 'RAW':
        raise ValueError('This native-observation audit requires a RAW reference experiment')
    config.update(project_root=str(root), training=training, experiment=experiment,
                  data_dir=str((root / training['data_dir']).resolve()),
                  size=experiment.get('size', 128))
    for key in ('catalogue', 'output_dir'):
        config[key] = str((root / config[key]).resolve())
    legacy = dict(experiment.get('legacy', training.get('legacy', {})))
    if legacy.get('raw_root'):
        legacy['raw_root'] = str((root / legacy['raw_root']).resolve())
    config['legacy'] = legacy
    # Refuse to write analysis exports inside an observational source tree.
    out = Path(config['output_dir'])
    for source in [Path(config['data_dir']), Path(legacy.get('raw_root', config['data_dir']))]:
        if out == source or source in out.parents:
            raise ValueError('Analysis output must be outside observational data directories')
    return config


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(config):
    """Union of catalogue and all RAW/CHANDRA categories, not a task cohort.

    All paths and copies remain in the manifest. Conflicting copies are flagged
    and not selected for preprocessing; duplicate category membership is normal.
    """
    root = Path(config['data_dir'])
    if not root.is_dir():
        raise FileNotFoundError(root)
    catalogue = (pd.read_csv(config['catalogue'], keep_default_na=False)
                 if Path(config['catalogue']).is_file() else pd.DataFrame(columns=['slug']))
    catalogue['cluster_id'] = catalogue['slug'].astype(str).str.strip()
    records, files, issues = {}, [], []

    def ensure(cid):
        return records.setdefault(cid, dict(cluster_id=cid, labels=set(), raw_path=None,
                                             chandra_path=None, legacy_path=None))

    for cid in catalogue.cluster_id:
        ensure(cid)
    for cid in catalogue.loc[catalogue.cluster_id.duplicated(False), 'cluster_id'].unique():
        issues.append(dict(cluster_id=cid, issue='duplicate catalogue identifier', status='confirmed'))
    for dirname, modality, suffix in [('RAW', 'radio', ''), ('CHANDRA', 'xray', 'CHANDRA')]:
        for path in sorted((root / dirname).glob('*/*.fits')):
            cid = path.stem.removesuffix(suffix) if suffix else path.stem
            info = ensure(cid)
            if modality == 'radio':
                info['labels'].add(path.parent.name)
            try:
                digest = _digest(path)
                error = ''
            except OSError as exc:
                digest, error = '', str(exc)
            files.append(dict(cluster_id=cid, modality=modality, path=str(path.resolve()),
                              category=path.parent.name, sha256=digest, discovery_error=error))
    legacy = config['legacy']
    if legacy.get('raw_root'):
        # Matches the project's cluster-directory optical convention, including orphans.
        for directory in sorted(Path(legacy['raw_root']).glob('*')):
            if not directory.is_dir():
                continue
            cid = directory.name
            path = directory / legacy.get('filename_pattern', '{cluster_id}_legacy.fits').format(cluster_id=cid)
            if path.is_file():
                ensure(cid)
                try:
                    digest, error = _digest(path), ''
                except OSError as exc:
                    digest, error = '', str(exc)
                files.append(dict(cluster_id=cid, modality='optical', path=str(path.resolve()),
                                  category='', sha256=digest, discovery_error=error))
    manifest = pd.DataFrame(files, columns=['cluster_id', 'modality', 'path', 'category', 'sha256', 'discovery_error'])
    keys = dict(radio='raw_path', xray='chandra_path', optical='legacy_path')
    for (cid, modality), group in manifest.groupby(['cluster_id', 'modality']):
        conflict = group.sha256.nunique() > 1 or group.discovery_error.ne('').any()
        if len(group) > 1:
            issues.append(dict(cluster_id=cid, issue=f'{modality}: {len(group)} source copies' +
                               (' with conflicting/unreadable contents' if conflict else ' (identical)'),
                               status='confirmed'))
        if not conflict:
            records[cid][keys[modality]] = group.iloc[0].path
    rows = []
    for cid, info in sorted(records.items()):
        row = dict(cluster_id=cid, annotations=';'.join(sorted(info['labels'])),
                   training_annotations=';'.join(sorted(info['labels'] & set(ANNOTATION_NAMES))),
                   has_annotation=bool(info['labels'] - {'unclassified'}))
        for modality, key in keys.items():
            row[modality] = bool(info[key])
            row[modality + '_file_present'] = bool(((manifest.cluster_id == cid) & (manifest.modality == modality)).any())
        rows.append(row)
    clusters = pd.DataFrame(rows).merge(catalogue.drop_duplicates('cluster_id'), on='cluster_id', how='left')
    for name in ('z', 'M500', 'M500_err', 'r500', 'r500_err'):
        if name in clusters:
            clusters[name] = pd.to_numeric(clusters[name], errors='coerce')
    return records, clusters, manifest, pd.DataFrame(issues, columns=['cluster_id', 'issue', 'status'])


def pixel_statistics(image):
    a = np.asarray(image)
    finite = a[np.isfinite(a)]
    result = dict(pixels=a.size, nan_fraction=float(np.isnan(a).mean()),
                  inf_fraction=float(np.isinf(a).mean()), zero_fraction=float((a == 0).mean()),
                  finite_fraction=float(np.isfinite(a).mean()))
    for label, value in zip(['min', 'p01', 'p05', 'median', 'p95', 'p99', 'max'],
                            np.percentile(finite, [0, 1, 5, 50, 95, 99, 100]) if finite.size else [np.nan]*7):
        result[label] = float(value)
    result['p99_minus_p01'] = result['p99'] - result['p01']
    # A robust span, not peak/noise dynamic range: no source-free region is guaranteed.
    return result


def inspect_file(path, modality):
    loader = io.load_optical_fits if modality == 'optical' else io.load_celestial_fits
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        image, header, wcs = loader(path)
    planes = image if image.ndim == 3 else image[None]
    scales = (proj_plane_pixel_scales(wcs) * u.deg).to_value(u.arcsec)
    ra, dec = wcs.pixel_to_world_values((image.shape[-1]-1)/2, (image.shape[-2]-1)/2)
    rows = []
    for i, plane in enumerate(planes):
        row = dict(status='ok', error='', warnings=' | '.join(str(x.message) for x in caught),
                   band=str(header.get(f'BAND{i}', i)) if modality == 'optical' else '',
                   height=plane.shape[0], width=plane.shape[1], pixel_x_arcsec=scales[0], pixel_y_arcsec=scales[1],
                   fov_x_arcmin=scales[0]*plane.shape[1]/60, fov_y_arcmin=scales[1]*plane.shape[0]/60,
                   image_center_ra_deg=float(ra), image_center_dec_deg=float(dec),
                   unit=str(header.get('BUNIT', 'unknown')), telescope=str(header.get('TELESCOP', 'unknown')),
                   instrument=str(header.get('INSTRUME', 'unknown')),
                   product=str(header.get('HDUCLAS2', header.get('CONTENT', 'unknown'))))
        # Preserve raw header values and keyword provenance; no inferred band/exposure units.
        for key in ('BMAJ', 'BMIN', 'BPA', 'EXPOSURE', 'EXPTIME', 'RESTFRQ', 'RESTFREQ',
                    'BANDWID', 'E_MIN', 'E_MAX', 'EUNIT', 'TIMEUNIT', 'FILTER'):
            row['header_' + key] = header.get(key, None)
        for axis in range(1, int(header.get('NAXIS', 2))+1):
            if str(header.get(f'CTYPE{axis}', '')).upper().startswith('FREQ'):
                row['frequency_value'] = header.get(f'CRVAL{axis}')
                row['frequency_unit'] = header.get(f'CUNIT{axis}', 'unknown')
                row['frequency_source'] = f'CRVAL{axis}'
        row.update(pixel_statistics(plane))
        if not np.isfinite(plane).any():
            row.update(status='invalid', error='No finite pixels')
        rows.append(row)
    return rows


def collect_observations(manifest, config):
    cache_dir = Path(config['output_dir']) / 'cache' / 'metadata'
    code_hash = hashlib.sha256((inspect.getsource(inspect_file) + inspect.getsource(pixel_statistics) +
                                inspect.getsource(io)).encode()).hexdigest()
    rows = []
    for file in tqdm(manifest.to_dict('records'), desc='FITS metadata and pixels'):
        token = hashlib.sha256((file['sha256']+file['modality']+code_hash).encode()).hexdigest()
        cache_path = cache_dir / (token + '.json')
        if config['cache'] and cache_path.exists() and not file['discovery_error']:
            measurements = json.loads(cache_path.read_text())
        else:
            try:
                measurements = inspect_file(file['path'], file['modality'])
            except Exception as exc:
                measurements = [dict(status='unreadable', error=f'{type(exc).__name__}: {exc}', band='')]
            if config['cache'] and measurements[0]['status'] != 'unreadable':
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(measurements, default=str))
        rows.extend({**file, **m} for m in measurements)
    return pd.DataFrame(rows)


def unique_observations(observations):
    # Identical copies across annotation directories are one observational product.
    return observations.drop_duplicates(['cluster_id', 'modality', 'sha256', 'band'])


def classification_tables(records):
    rows = []
    for cid, info in records.items():
        # Exactly the annotation vocabulary discovered by the training loader.
        labels = info['labels'] & set(ANNOTATION_NAMES)
        for policy in ('positive', 'ignore'):
            for task in ('de_nde', 'rr_binary', 'rh_binary', 'rr_rh'):
                row = dict(cluster_id=cid, task=task, candidate_policy=policy,
                           eligible=False, reason='', target_0=np.nan, target_1=np.nan,
                           valid_0=False, valid_1=False)
                try:
                    if not info['raw_path']:
                        raise ValueError('No unambiguous RAW observation')
                    target, mask = task_target({'labels': labels}, task, policy, ('DE', 'NDE'))
                    row['eligible'] = True
                    if task == 'rr_rh':
                        row.update(target_0=float(target[0]), target_1=float(target[1]),
                                   valid_0=bool(mask[0]), valid_1=bool(mask[1]))
                    else:
                        row.update(target_0=int(target), valid_0=True)
                except ValueError as exc:
                    row['reason'] = str(exc)
                rows.append(row)
    targets = pd.DataFrame(rows)
    summary = []
    for (task, policy), group in targets.groupby(['task', 'candidate_policy']):
        for index, name in enumerate(('RR', 'RH') if task == 'rr_rh' else
                                     ('NDE (target 1; DE is 0)',) if task == 'de_nde' else (task[:2].upper(),)):
            valid = group.eligible & group[f'valid_{index}']
            positives = int((group.loc[valid, f'target_{index}'] == 1).sum())
            negatives = int(valid.sum()) - positives
            summary.append(dict(task=task, candidate_policy=policy, target=name,
                                eligible=int(group.eligible.sum()), excluded=int((~group.eligible).sum()),
                                masked=int((group.eligible & ~group[f'valid_{index}']).sum()),
                                positives=positives, negatives=negatives,
                                positive_fraction=positives / valid.sum() if valid.any() else np.nan))
    return targets, pd.DataFrame(summary)


def aligned_sample(info, config):
    """Use actual preprocessing; cache only under the analysis output directory."""
    paths = [info[k] for k in ('raw_path', 'chandra_path') if info.get(k)]
    token = hashlib.sha256(json.dumps([[_digest(p) for p in paths], config['size'],
                                       inspect.getsource(preprocessing), inspect.getsource(io)]).encode()).hexdigest()
    path = Path(config['output_dir']) / 'cache' / 'aligned' / (token + '.npz')
    if config['cache'] and path.exists():
        from astropy.wcs import WCS
        from astropy.io.fits import Header
        with np.load(path, allow_pickle=False) as data:
            result = {k: data[k].copy() for k in data.files if k != 'wcs_header'}
            result['wcs'] = WCS(Header.fromstring(str(data['wcs_header']), sep='\n'))
        return result
    result = preprocessing.preprocess(info, 'pair' if info.get('chandra_path') else 'raw', config['size'])
    if config['cache']:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: v for k, v in result.items() if k not in ('wcs', 'radio_metadata')},
                            wcs_header=result['wcs'].to_header().tostring(sep='\n'))
    return result


def collect_alignment(records, config):
    rows = []
    if not config['alignment']:
        return pd.DataFrame(columns=['cluster_id', 'status', 'coverage', 'error'])
    for cid, info in tqdm(records.items(), desc='Pipeline preprocessing'):
        if not info['raw_path']:
            continue
        try:
            result = aligned_sample(info, config)
            row = dict(cluster_id=cid, status='ok', error='', factor=int(result['factor']),
                       coverage=float(result['chandra_mask'].mean()) if 'chandra_mask' in result else np.nan)
            for key in ('raw', 'chandra'):
                if key in result:
                    mask = result.get('chandra_mask') if key == 'chandra' else None
                    for stage, a in [('reduced', result[key]), ('normalized', preprocessing.normalize(result[key], mask))]:
                        for name, value in pixel_statistics(a if mask is None else a[mask]).items() if (mask is None or mask.any()) else []:
                            row[f'{key}_{stage}_{name}'] = value
            rows.append(row)
        except Exception as exc:
            rows.append(dict(cluster_id=cid, status='error', error=f'{type(exc).__name__}: {exc}', coverage=np.nan))
    return pd.DataFrame(rows)


def label_matrix(clusters):
    labels = sorted({label for text in clusters.annotations for label in text.split(';') if label})
    return pd.DataFrame({label: clusters.annotations.map(lambda s: label in s.split(';')).astype(int)
                         for label in labels}, index=clusters.index)


def diagnostic_table(clusters, observations, alignment, initial, config):
    issues = initial.to_dict('records')
    def add(cid, issue, status='potential — manual inspection'):
        issues.append(dict(cluster_id=cid, issue=issue, status=status))
    for row in observations.to_dict('records'):
        if row['status'] != 'ok':
            add(row['cluster_id'], row['modality'] + ': ' + row.get('error', ''), 'confirmed')
        elif row.get('finite_fraction', 1) < 1:
            add(row['cluster_id'], row['modality'] + ': non-finite pixels (may be intended footprint)', 'confirmed')
        if row.get('warnings'):
            add(row['cluster_id'], row['modality'] + ': FITS/WCS warning: ' + row['warnings'])
    for row in clusters.itertuples():
        if not row.radio_file_present:
            add(row.cluster_id, 'No RAW file in configured classified tree', 'confirmed')
        if not row.annotations:
            add(row.cluster_id, 'No RAW directory annotation', 'confirmed')
    for row in alignment.to_dict('records'):
        if row['status'] != 'ok':
            add(row['cluster_id'], 'Preprocessing failed: ' + row['error'], 'confirmed')
        elif row['coverage'] < config['coverage_threshold']:
            add(row['cluster_id'], f"Reduced-grid X-ray coverage {row['coverage']:.3f} below configured threshold")
    for _, group in observations[observations.status == 'ok'].groupby(['modality', 'unit', 'band']):
        for column in ('pixels', 'p99_minus_p01'):
            values = pd.to_numeric(group[column], errors='coerce')
            q1, q3 = values.quantile([.25, .75])
            if q3 > q1:
                for cid in group.loc[(values < q1-3*(q3-q1)) | (values > q3+3*(q3-q1)), 'cluster_id']:
                    add(cid, f'{group.iloc[0].modality}: extreme {column} (3 IQR rule, within unit group)')
    return pd.DataFrame(issues, columns=['cluster_id', 'issue', 'status']).drop_duplicates()


class Report:
    def __init__(self, config):
        self.config, self.figures, self.tables = config, [], {}
        plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                             'savefig.dpi': config['dpi'], 'figure.constrained_layout.use': True,
                             'pdf.fonttype': 42, 'ps.fonttype': 42})

    def table(self, name, frame):
        self.tables[name] = frame
        if self.config['export']:
            folder = Path(self.config['output_dir']) / 'tables'
            folder.mkdir(parents=True, exist_ok=True)
            frame.to_csv(folder / f'{name}.csv', index=False)
            (folder / f'{name}.tex').write_text(frame.to_latex(index=False, escape=True, float_format='%.4g'))
        return frame

    def figure(self, name, fig):
        self.figures.append(name)
        if self.config['export']:
            folder = Path(self.config['output_dir']) / 'figures'
            folder.mkdir(parents=True, exist_ok=True)
            for extension in ('pdf', 'png'):
                fig.savefig(folder / f'{name}.{extension}', bbox_inches='tight', dpi=self.config['dpi'])
        plt.show()
        plt.close(fig)

    def histograms(self, name, frame, columns):
        fig, axes = plt.subplots(1, len(columns), figsize=(4*len(columns), 3.3), squeeze=False)
        for ax, (column, label) in zip(axes[0], columns.items()):
            values = pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors='coerce').dropna()
            if len(values):
                ax.hist(values, bins=min(25, max(5, int(np.sqrt(len(values))))), color='#346b8a')
            else:
                ax.text(.5, .5, 'Metadata unavailable', ha='center', transform=ax.transAxes)
            ax.set(xlabel=label, ylabel='Observations')
        self.figure(name, fig)


def observation_figures(observations, modality, report):
    frame = observations[(observations.modality == modality) & (observations.status == 'ok')]
    report.histograms(f'{modality}_geometry', frame, {'width': 'Width [pixels]', 'height': 'Height [pixels]',
                                                      'pixel_x_arcsec': 'Projected x pixel scale [arcsec]'})
    if modality == 'radio':
        beam = frame.copy()
        for axis in ('BMAJ', 'BMIN'):
            beam[axis] = pd.to_numeric(beam.get('header_'+axis), errors='coerce') * 3600
        report.histograms('radio_beams', beam, {'BMAJ': 'Beam major FWHM [arcsec]', 'BMIN': 'Beam minor FWHM [arcsec]'})
    if modality == 'xray':
        # EXPOSURE/EXPTIME are retained as header quantities, not silently assigned seconds.
        report.histograms('xray_exposure', frame, {'header_EXPOSURE': 'EXPOSURE [header units; see TIMEUNIT]',
                                                   'header_EXPTIME': 'EXPTIME [header units; see TIMEUNIT]'})
    for (unit, band), group in frame.groupby(['unit', 'band'], dropna=False):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for column, label in [('p01', 'Per-image p01'), ('median', 'Per-image median'), ('p99', 'Per-image p99')]:
            ax.hist(group[column].dropna(), bins=25, histtype='step', label=label)
        ax.set(xlabel=f'Intensity [{unit}] — band {band or "not specified"}', ylabel='Observations')
        ax.set_xscale('symlog', linthresh=max(float(group.p99_minus_p01.median())*.01, 1e-12))
        ax.legend()
        safe = hashlib.sha256(f'{unit}/{band}'.encode()).hexdigest()[:8]
        report.figure(f'{modality}_intensities_{safe}', fig)


def select_gallery(records, config):
    if config['gallery_ids']:
        unknown = set(config['gallery_ids']) - records.keys()
        if unknown:
            raise ValueError(f'Unknown gallery IDs: {sorted(unknown)}')
        return config['gallery_ids'][:config['gallery_limit']]
    selected = []
    for category in config['gallery_categories']:
        candidates = [cid for cid, info in sorted(records.items()) if category in info['labels'] and info['raw_path']]
        candidates.sort(key=lambda cid: (not bool(records[cid]['chandra_path']), cid))
        selected.extend([cid for cid in candidates if cid not in selected][:1])
    return selected[:config['gallery_limit']]


def _image(ax, array, title, unit='per-image display stretch'):
    array = np.asarray(array)
    mask = np.isfinite(array)
    shown = np.ma.array(preprocessing.normalize(array), mask=~mask)
    artist = ax.imshow(shown, origin='lower', cmap='magma', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    ax.figure.colorbar(artist, ax=ax, fraction=.046, pad=.04, label=unit)


def gallery(records, config, report):
    if not config['galleries']:
        return
    for cid in select_gallery(records, config):
        info = records[cid]
        fig = plt.figure(figsize=(13, 4))
        try:
            aligned = aligned_sample(info, config)
            ax = fig.add_subplot(131, projection=aligned['wcs'])
            _image(ax, aligned['raw'], f'{cid}\nRadio: {", ".join(sorted(info["labels"]))}')
            ax.set(xlabel='RA', ylabel='Dec')
            if 'chandra' in aligned:
                finite = aligned['chandra'][aligned['chandra_mask']]
                if finite.size and np.ptp(finite) > 0:
                    levels = np.unique(np.percentile(finite, [80, 95, 99]))
                    ax.contour(np.ma.masked_invalid(aligned['chandra']), levels=levels, colors='cyan', linewidths=.6)
                ax = fig.add_subplot(132, projection=aligned['wcs'])
                _image(ax, aligned['chandra'], f'X-ray; valid coverage {aligned["chandra_mask"].mean():.1%}')
                ax.set(xlabel='RA', ylabel='Dec')
            else:
                ax = fig.add_subplot(132); ax.text(.5, .5, 'X-ray unavailable', ha='center'); ax.set_axis_off()
            if info['legacy_path']:
                optical = preprocessing.preprocess_optical(info, config['size'], bands=config['legacy'].get('bands'))
                ax = fig.add_subplot(133, projection=optical['wcs'])
                band = config['legacy'].get('bands', ['first'])[0]
                _image(ax, optical['legacy'][0], f'Optical band {band}; aligned to radio')
                ax.set(xlabel='RA', ylabel='Dec')
            else:
                ax = fig.add_subplot(133); ax.text(.5, .5, 'Optical unavailable', ha='center'); ax.set_axis_off()
        except Exception as exc:
            fig.text(.5, .5, f'{cid}: {exc}', ha='center', wrap=True)
        report.figure('gallery_' + cid, fig)


def preprocessing_example(records, config, report):
    if not config['galleries']:
        return
    import torch
    choices = [cid for cid in select_gallery(records, config) if records[cid]['raw_path']]
    if not choices:
        return
    cid = choices[0]
    info = records[cid]
    result = aligned_sample(info, config)
    raw, _, radio_wcs = io.load_celestial_fits(info['raw_path'])
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(231, projection=radio_wcs); _image(ax, raw, 'Original radio (display stretch)')
    if info['chandra_path']:
        xray, _, xwcs = io.load_celestial_fits(info['chandra_path'])
        ax = fig.add_subplot(232, projection=xwcs); _image(ax, xray, 'Original X-ray (display stretch)')
    factor = int(result['factor'])
    window = preprocessing.centered_raw_window(raw.shape, config['size'], factor)
    ax = fig.add_subplot(233, projection=radio_wcs.slice(window)); _image(ax, raw[window], 'Exact pipeline radio crop')
    ax = fig.add_subplot(234, projection=result['wcs']); _image(ax, result['raw'], f'Block mean; factor {factor}')
    normalized = preprocessing.normalize(result['raw'])
    ax = fig.add_subplot(235); artist = ax.imshow(normalized, origin='lower', vmin=0, vmax=1, cmap='magma')
    ax.set_title('Actual normalized radio'); fig.colorbar(artist, ax=ax, label='Normalized intensity')
    if 'chandra_mask' in result:
        ax = fig.add_subplot(236); artist = ax.imshow(result['chandra_mask'], origin='lower', vmin=0, vmax=1, cmap='gray')
        ax.set_title('Validity after reprojection + block mean'); fig.colorbar(artist, ax=ax, label='Valid pixel')
    report.figure('preprocessing_' + cid, fig)
    tensor = torch.from_numpy(normalized[None])
    if 'chandra' in result:
        tensor = torch.from_numpy(np.stack([normalized, preprocessing.normalize(result['chandra'], result['chandra_mask']),
                                           result['chandra_mask'].astype(np.float32)]))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, (angle, flip) in zip(axes, [(0, False), (30, False), (90, True), (150, True)]):
        augmented = transform_aligned(tensor, angle=angle, flip=flip)
        ax.imshow(augmented[0], origin='lower', cmap='magma', vmin=0, vmax=1)
        ax.set_title(f'{angle}°; flip={flip}'); ax.set_axis_off()
    report.figure('augmentation_' + cid, fig)


def finish(report, clusters, observations, tasks, diagnostics, alignment):
    config = report.config
    summary = dict(unique_clusters=len(clusters),
                   selected_file_availability={m: int(clusters[m].sum()) for m in MODALITIES},
                   clusters_with_directory_membership=int(clusters.annotations.ne('').sum()),
                   clusters_with_original_annotations=int(clusters.has_annotation.sum()),
                   unique_observational_products=int(observations.drop_duplicates(['cluster_id', 'modality', 'sha256']).shape[0]),
                   invalid_or_unreadable_products=int(observations.loc[observations.status != 'ok'].drop_duplicates('path').shape[0]),
                   alignment_attempted=len(alignment), alignment_success=int(alignment.status.eq('ok').sum()),
                   diagnostic_rows=len(diagnostics), figures=report.figures,
                   classification_tasks=json.loads(tasks.to_json(orient='records')))
    text = (f"The configured catalogue and observational directories contain {len(clusters)} unique cluster identifiers. "
            f"Unambiguous radio, X-ray and optical source files are available for {int(clusters.radio.sum())}, "
            f"{int(clusters.xray.sum())} and {int(clusters.optical.sum())} clusters, respectively. "
            f"Original directory annotations other than 'unclassified' are present for {summary['clusters_with_original_annotations']} clusters. "
            f"The audit identified {summary['invalid_or_unreadable_products']} invalid or unreadable products. "
            "These are pre-split counts; augmented variants are not independent clusters. "
            "Original annotations and pipeline targets are tabulated separately, including candidate-policy sensitivity. "
            "X-ray validity describes finite support after reprojection and block averaging, not physical zero emission. "
            "The sample is defined by the configured catalogue and local files and is not asserted to be survey-complete.")
    if config['export']:
        folder = Path(config['output_dir']); folder.mkdir(parents=True, exist_ok=True)
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
        (folder / 'dataset_summary.txt').write_text(text + '\n')
        (folder / 'configuration.json').write_text(json.dumps(config, indent=2))
        import importlib.metadata
        from . import labels, augmentation
        versions = {name: importlib.metadata.version(name) for name in
                    ('numpy', 'pandas', 'astropy', 'reproject', 'matplotlib', 'torch', 'clusterprep')}
        (folder / 'provenance.json').write_text(json.dumps(dict(python=platform.python_version(), packages=versions,
            code_sha256={m.__name__: hashlib.sha256(inspect.getsource(m).encode()).hexdigest()
                         for m in (io, preprocessing, labels, augmentation)},
            analysis_sha256=_digest(__file__), manifest='tables/file_manifest.csv'), indent=2))
    return summary, text
