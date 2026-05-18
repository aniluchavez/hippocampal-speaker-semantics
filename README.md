# Neural Encoding Analysis

Poisson ridge regression pipeline for neural encoding of speech in intracranial recordings.

## Package structure

```
neural_encoding/
├── regression.py       # Design-matrix building, Poisson ridge regression, alpha CV, null models
├── reliability.py      # Beta reliability, noise ceiling, split-half analysis
└── spike_processing.py # Spike loading (.mat), region cell selection, event extraction, binning
```

## Usage in notebooks / scripts

```python
from neural_encoding import regression as reg
from neural_encoding import reliability as rel
from neural_encoding import spike_processing as spu
```

## Scripts

| Script | Purpose |
|---|---|
| `scripts/run_spike_pipeline.py` | End-to-end spike extraction per patient |
| `scripts/sweep_regression.py` | Regression sweep across parameter grids |
| `scripts/runregreshpercluster.py` | Cluster-submission wrapper |

## Utilities

| File | Purpose |
|---|---|
| `utils/build_XY.py` | Build clean X (embeddings) and Y (spikes) matrices |
| `utils/extract_spikes_window.py` | Windowed spike extraction |
| `utils/poisson_ridge_torch.py` | PyTorch Poisson ridge backend |
| `utils/regression_wrappers.py` | Higher-level regression wrappers |
| `utils/inputwrappers.py` | Input data wrappers |
| `utils/poisson_xy_utils.py` | XY utility functions |

## Legacy

Older versions of the core modules are preserved in `legacy/` for reference.

## Install

```bash
pip install -r requirements.txt
```
