# Dataset analysis

Open [dataset_analysis.ipynb](../notebooks/dataset_analysis.ipynb) and run all cells
with the workspace Python environment. Install the package's analysis extra if
needed (`pip install -e '.[analysis]'` from the package directory).

The notebook finds the workspace from its current directory and reads
[the JSON configuration](../configs/dataset_analysis.json). It uses the first
experiment in the referenced training configuration for data paths, optical
sources, output image size and augmentation settings. The supplied configuration
references the current RAW + Legacy + CHANDRA experiment. Run from the workspace
or a directory beneath it; no absolute personal paths are embedded in the notebook.

The ten sections cover the complete pre-split inventory, radio, X-ray, optical,
cluster metadata, labels/tasks, preprocessing, quality, galleries and report
summary. Dataset counts use the union of the catalogue and discovered cluster
identifiers. Directory annotations, catalogue classifications and pipeline
targets remain separate. The training loader's supported label vocabulary is
applied only when calculating targets. Conflicting duplicate source files are
retained in the manifest and flagged, not arbitrarily used for preprocessing.

Outputs default to `outputs/dataset_analysis/` relative to the workspace:

- `figures/`: PDF and 300-dpi PNG figures.
- `tables/`: CSV and escaped LaTeX tables (LaTeX requires `booktabs`).
- `summary.json` and `dataset_summary.txt`: computed report summaries.
- `configuration.json` and `provenance.json`: reproducibility information.
- `cache/`: content-keyed FITS statistics and aligned arrays, isolated from training caches.

Use `export: false` for interactive analysis without report writes. Also use
`cache: false` to disable analysis-cache writes/reads. `alignment: false` skips
the population-wide preprocessing audit; `galleries: false` skips image galleries
and example preprocessing. Such runs explicitly have no population coverage
measurements. `gallery_ids`, `gallery_categories`, and `gallery_limit` control
representative selection. Full alignment is enabled by default and may take
several minutes on the first run. Source checksums invalidate stale cached data.
Report files are regenerated on each export-enabled run; source data are never
modified. No model training or trained checkpoints are needed.

Scientific limitations are stated in the notebook: approximate projected angular
sizes, header-only observational quantities, per-image rather than pooled-pixel
intensity distributions, finite-support rather than exposure coverage, candidate
uncertainty, and optical display stretches distinct from train-fitted normalization.
The source CSV does not provide physical-property units or catalogue RA/Dec;
plots label WCS image-centre coordinates explicitly. The audit does not estimate
noise without a defensible region or metadata, or derive physical fluxes.

Implementation lives in [dataset_analysis.py](../src/clusterprep/dataset_analysis.py).
Run its targeted checks with `python -m pytest tests/test_dataset_analysis.py -q`
from the package directory.
