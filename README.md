# Wafer Map Similarity Search

This project compares two ways to retrieve historical wafer maps with similar
failure patterns from the WM-811K dataset:

1. spatial features designed by hand, followed by nearest-neighbor search;
2. embeddings learned by a convolutional autoencoder, followed by the same search.

The main metric is per-class precision@5. The project also tests whether distance
to the nearest known wafer can flag a defect class that the model never saw.

## Current status

M0 establishes the project structure, dependencies, fixed random seeds, tests,
and software-version logging. Later milestones will add data preparation,
evaluation, retrieval models, out-of-distribution detection, and final figures.

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
