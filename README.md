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
- M2: stratified fit/dev/test splits, capped test-query sampling, per-class
  precision@5, and a random-retrieval sanity check.
- M3: standardized handcrafted spatial features and nearest-neighbor baseline
  retrieval, evaluated on the development split.
- M4: categorical convolutional autoencoder, one-batch overfit verification,
  balanced full training, early stopping, and deterministic 64-number embeddings.

Later milestones will add learned retrieval evaluation, the final comparison,
out-of-distribution detection, and portfolio figures.

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

## Verify the evaluation harness

```bash
python scripts/check_random_retriever.py
```

This creates a stratified split, samples at most 2,000 test queries per class,
and confirms that random precision@5 is consistent with each class's frequency
in the training database. The generated comparison is saved to
`results/m2_random_sanity.csv`.

## Run the handcrafted baseline

```bash
python scripts/run_baseline_dev.py
```

The first run extracts 36 spatial features for every processed wafer and caches
them locally. The script fits its standardizer and nearest-neighbor database on
fit rows only, uses development rows as queries, and saves per-class precision@5
to `results/m3_baseline_dev.csv`. Test rows remain untouched during feature
design.

## Train the convolutional autoencoder

```bash
python scripts/check_autoencoder_overfit.py
MPLCONFIGDIR=.cache/matplotlib python scripts/train_autoencoder.py
```

The first command verifies that the model can memorize one mixed batch. Full
training uses all fit-split defect wafers and 5,000 sampled `none` wafers, while
the separate development split controls checkpoint selection. Checkpoints stay
local under `checkpoints/`; the training history, summary, and learning curves
are reproducible tracked outputs.

![Autoencoder training and validation curves](figures/m4_training_curves.png)
