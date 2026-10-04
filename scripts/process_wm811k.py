"""Create the compact M1 cache from the raw WM-811K pickle."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from wafer_retrieval.data import process_labeled_maps


CLASS_ORDER = (
    "none",
    "Center",
    "Donut",
    "Edge-Ring",
    "Edge-Loc",
    "Loc",
    "Scratch",
    "Random",
    "Near-full",
)

REFERENCE_COUNTS = {
    "none": 147_431,
    "Edge-Ring": 9_680,
    "Edge-Loc": 5_189,
    "Center": 4_294,
    "Loc": 3_593,
    "Scratch": 1_193,
    "Random": 866,
    "Donut": 555,
    "Near-full": 149,
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=project_root / "LSWMD.pkl")
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "data" / "processed"
    )
    parser.add_argument(
        "--figure", type=Path, default=project_root / "figures" / "m1_resize_examples.png"
    )
    parser.add_argument(
        "--summary", type=Path, default=project_root / "results" / "m1_summary.json"
    )
    return parser.parse_args()


def check_class_counts(counts: Counter[str]) -> None:
    found_classes = set(counts)
    expected_classes = set(CLASS_ORDER)
    if found_classes != expected_classes:
        raise ValueError(
            f"Expected classes {sorted(expected_classes)}, found {sorted(found_classes)}"
        )

    for label, reference in REFERENCE_COUNTS.items():
        relative_difference = abs(counts[label] - reference) / reference
        if relative_difference > 0.01:
            warnings.warn(
                f"{label} count {counts[label]:,} differs from reference "
                f"{reference:,} by {relative_difference:.1%}",
                stacklevel=2,
            )


def save_example_figure(
    examples: list[tuple[str, np.ndarray, np.ndarray]], figure_path: Path
) -> None:
    colors = ListedColormap(["#f2f2f2", "#4c78a8", "#e45756"])
    figure, axes = plt.subplots(len(examples), 2, figsize=(6, 18))

    for row, (label, original, resized) in enumerate(examples):
        axes[row, 0].imshow(original, cmap=colors, vmin=0, vmax=2, interpolation="nearest")
        axes[row, 0].set_title(f"{label}: original {original.shape[0]}×{original.shape[1]}")
        axes[row, 1].imshow(resized, cmap=colors, vmin=0, vmax=2, interpolation="nearest")
        axes[row, 1].set_title(f"{label}: resized 32×32")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

    figure.suptitle("WM-811K categorical resize check", fontsize=14)
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    started = time.time()
    print(f"Loading {args.raw} ...", flush=True)
    dataframe = pd.read_pickle(args.raw)
    total_rows = len(dataframe)
    print(f"Loaded {total_rows:,} rows. Cleaning labeled wafers ...", flush=True)

    raw_maps = dataframe["waferMap"].to_numpy()
    raw_labels = dataframe["failureType"].to_numpy()
    maps, labels, source_indices = process_labeled_maps(raw_maps, raw_labels)

    counts = Counter(labels.tolist())
    check_class_counts(counts)

    examples: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label in CLASS_ORDER:
        processed_index = int(np.flatnonzero(labels == label)[0])
        source_index = int(source_indices[processed_index])
        examples.append((label, np.asarray(raw_maps[source_index]), maps[processed_index]))

    # The compact arrays and nine examples are now independent of the DataFrame.
    del dataframe, raw_maps, raw_labels

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "maps_32.npy", maps)
    np.save(args.output_dir / "labels.npy", labels)
    np.save(args.output_dir / "source_indices.npy", source_indices)

    summary = {
        "source_rows": total_rows,
        "labeled_rows": int(len(labels)),
        "maps_shape": list(maps.shape),
        "maps_dtype": str(maps.dtype),
        "maps_values": np.unique(maps).tolist(),
        "class_counts": {label: counts[label] for label in CLASS_ORDER},
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save_example_figure(examples, args.figure)

    print(json.dumps(summary, indent=2))
    print(f"Saved cache to {args.output_dir}")
    print(f"Saved resize check to {args.figure}")
    print(f"Completed in {time.time() - started:.1f} seconds")


if __name__ == "__main__":
    main()
