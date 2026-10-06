"""Draw the M5 grouped precision chart in a plotting-only process."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
METHODS = ("random", "baseline", "autoencoder")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "results/m5_comparison.csv")
    parser.add_argument("--output", type=Path, default=root / "figures/m5_comparison.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8") as source:
        rows = {row["class"]: row for row in csv.DictReader(source)}

    x = np.arange(len(CLASS_ORDER))
    width = 0.25
    colors = {"random": "#9ca3af", "baseline": "#2563eb", "autoencoder": "#f97316"}
    figure, axis = plt.subplots(figsize=(14, 5.5))
    for offset, method in enumerate(METHODS):
        axis.bar(
            x + (offset - 1) * width,
            [float(rows[label][f"{method}_mean"]) for label in CLASS_ORDER],
            width,
            yerr=[float(rows[label][f"{method}_std"]) for label in CLASS_ORDER],
            capsize=2,
            label=method.title(),
            color=colors[method],
        )
    axis.set(
        ylabel="Precision@5 (mean across seeds)",
        xlabel="Query label",
        title="Wafer retrieval comparison over stratified test splits",
        ylim=(0, 1.05),
        xticks=x,
        xticklabels=CLASS_ORDER,
    )
    axis.tick_params(axis="x", rotation=35)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
