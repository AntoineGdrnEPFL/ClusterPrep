"""Execute the unified notebook on actual artifacts (no training or inference).

Run from an environment with clusterprep[analysis] installed:
python examples/execute_unified_analysis.py --root ../results --output /tmp/analysis-check
"""
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(package/'notebooks/analyse_trainings.ipynb', as_version=4)
    nb.cells[1].source += f'\nROOT_DIR = Path({str(args.root.resolve())!r})\nEXPORT_ROOT = Path({str(output)!r})\n'
    # Select representatives across tasks and transfer/CV regimes for detail figures.
    discovery = next(c for c in nb.cells if c.cell_type == 'code' and '# Appliquer :' in c.source)
    discovery.source += '''
representatives = {}
for run in runs:
    if run.status == 'completed':
        representatives.setdefault((run.task_type, run.stage, run.transfer_strategy), run)
detail_runs = list(representatives.values())[:8]
'''
    NotebookClient(nb, timeout=180, kernel_name='python3').execute(cwd=str(package/'notebooks'))
    nbformat.write(nb, output/'analyse_trainings.executed.ipynb')
    print(output/'analyse_trainings.executed.ipynb')


if __name__ == '__main__':
    main()
