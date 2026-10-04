# Wafer Map Similarity Search

This project compares two ways to retrieve historical wafer maps with similar
failure patterns from the WM-811K dataset:

1. spatial features designed by hand, followed by nearest-neighbor search;
2. embeddings learned by a convolutional autoencoder, followed by the same search.

The main metric is per-class precision@5. The project also tests whether distance
to the nearest known wafer can flag a defect class that the model never saw.

## Current status

- M0: project structure, dependencies, fixed random seeds, tests, and
  software-version logging.
- M1: labeled-wafer extraction, categorical 32×32 resizing, local array cache,
  class-count validation, and a visual resize check.

Later milestones will add evaluation, retrieval models, out-of-distribution
detection, and final figures.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
pytest
python scripts/check_setup.py
```

The raw `LSWMD.pkl` file belongs in the repository root but is intentionally
excluded from Git because it is large and must remain local.

## Prepare the labeled data

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/process_wm811k.py
```

The command keeps the 172,950 labeled wafers, resizes their maps without
blending the categorical values, and writes the local cache under
`data/processed/`. Raw data and processed arrays are excluded from Git.

![Original and resized examples for all nine classes](figures/m1_resize_examples.png)
