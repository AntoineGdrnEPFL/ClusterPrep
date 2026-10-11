"""Execute all analysis cells in a real kernel using this Python environment."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path, help='New directory; existing paths are refused')
    parser.add_argument('--run-dir', type=Path, action='append', default=[])
    parser.add_argument('--bootstrap-iterations', type=int, default=200)
    parser.add_argument('--threshold-method', choices=['stored','fixed','f1','recall_at_precision'], default='stored')
    args = parser.parse_args()
    import nbformat
    from nbclient import NotebookClient
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    notebook = nbformat.read(root/'notebooks'/'analyse_multilabel.ipynb', as_version=4)
    with tempfile.TemporaryDirectory(prefix='clusterprep-jupyter-') as temporary:
        temporary = Path(temporary)
        kernel = temporary/'kernels'/'clusterprep-analysis'
        kernel.mkdir(parents=True)
        (kernel/'kernel.json').write_text(json.dumps({
            'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'Clusterprep analysis', 'language':'python'}))
        os.environ['JUPYTER_PATH'] = str(temporary)
        for name in ['IPYTHONDIR','JUPYTER_RUNTIME_DIR','MPLCONFIGDIR','XDG_CACHE_HOME']:
            directory = temporary/name.lower()
            directory.mkdir()
            os.environ[name] = str(directory)
        os.environ['CLUSTERPREP_ANALYSIS_OUTPUT_ROOT'] = str(output/'exports')
        os.environ['CLUSTERPREP_BOOTSTRAP_ITERATIONS'] = str(args.bootstrap_iterations)
        os.environ['CLUSTERPREP_THRESHOLD_METHOD'] = args.threshold_method
        os.environ['CLUSTERPREP_ANALYSIS_RUN_DIRS'] = json.dumps([str(p.resolve()) for p in args.run_dir])
        client = NotebookClient(notebook, timeout=300, kernel_name='clusterprep-analysis',
                                resources={'metadata': {'path': str(root)}})
        try:
            client.execute()
        finally:
            nbformat.write(notebook, output/'analyse_multilabel.executed.ipynb')
    errors = [entry for cell in notebook.cells if cell.cell_type == 'code'
              for entry in cell.get('outputs',[]) if entry.output_type == 'error']
    summary = {'code_cells':sum(c.cell_type=='code' for c in notebook.cells),
               'executed_cells':sum(c.cell_type=='code' and c.execution_count is not None for c in notebook.cells),
               'errors':len(errors),'notebook':str(output/'analyse_multilabel.executed.ipynb')}
    (output/'execution.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
