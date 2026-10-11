"""Create a reproducible synthetic catalogue and exercise the real CLI.

Run from the workspace with PYTHONPATH=clusterprep/src python clusterprep/examples/audit_smoke.py OUTPUT.
No astronomical data or network access is required. Existing OUTPUT is refused.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def main():
    import matplotlib
    matplotlib.use("Agg")
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    data = root / 'data'
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crpix = [8, 8]
    wcs.wcs.crval = [180, 45]
    wcs.wcs.cdelt = [-.01, .01]
    rng = np.random.default_rng(7)
    patterns = [{'DE', 'RR'}, {'DE', 'RH'}, {'DE', 'RR', 'RH'}, {'cRR'}, {'cRH'}, {'NDE'}]
    for i in range(60):
        labels = patterns[i % len(patterns)]
        image = rng.normal(size=(16, 16)).astype('float32')
        for label in sorted(labels):
            directory = data / 'RAW' / label
            directory.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(image, wcs.to_header()).writeto(directory / f'cluster{i}.fits')
        if i % 4:
            directory = data / 'CHANDRA' / sorted(labels)[0]
            directory.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(image, wcs.to_header()).writeto(directory / f'cluster{i}CHANDRA.fits')
    base = dict(size=16, width=2, epochs=2, batch_size=4, device='cpu', augment=False,
                processed_dir=str(root/'processed'), verbose=False, mode='multimodal')
    experiments = []
    for task, model in [('de_nde','cnn'), ('rr_rh','cnn'), ('rr_rh','dual_ssn'), ('rr_rh','dual_encoder_ssn')]:
        experiments.append(dict(base, name=f'{task}-{model}', task=task, model=model,
            model_params={'L': 2, 'hidden_dim2': 2} if 'ssn' in model else {}))
    spec = dict(data_dir=str(data), output_dir=str(root/'runs'), experiments=experiments,
                comparison_cohort='all')
    config = root/'training_config.json'
    config.write_text(json.dumps(spec, indent=2))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    commands = [
        ['--list-models'],
        ['createconfig', '--output', str(root/'wizard_config.json')],
        ['preprocess', str(data), '--size', '16', '--mode', 'multimodal', '--class-names',
         'DE', 'NDE', 'RR', 'RH', 'cRR', 'cRH', '--processed-dir', str(root/'processed'), '--quiet'],
        [str(config)],
    ]
    for index, args in enumerate(commands):
        completed = subprocess.run([sys.executable, '-m', 'clusterprep', *args],
            input='\n' * 100 if args[0] == 'createconfig' else None,
            text=True, capture_output=True, env=env)
        (root/f'cli-{index}.log').write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f'CLI failed: {args}; see cli-{index}.log')
    from clusterprep.visualization import plot_history
    import matplotlib.pyplot as plt
    summary = []
    for run in sorted((root/'runs').iterdir()):
        if not run.is_dir():
            continue
        metrics = json.loads((run/'metrics.json').read_text())
        plt.close(plot_history(run, run/'history.png'))
        summary.append(dict(run=str(run), best_epoch=metrics['best_epoch'], epochs=metrics['epochs_ran'],
                            test_loss=metrics['test']['loss'], macro_f1=metrics['test']['macro_f1']))
    (root/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
