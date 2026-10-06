"""Draw the fixed query-and-neighbor examples from the saved demo manifest."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, default=root / "data/processed/maps_32.npy")
    parser.add_argument("--manifest", type=Path, default=root / "results/m7_demo.json")
    parser.add_argument("--figure", type=Path, default=root / "figures/m7_retrieval_demo.png")
    args = parser.parse_args()

    maps = np.load(args.maps, mmap_mode="r")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    colors = ListedColormap(["#243041", "#f1f5f9", "#e95050"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], colors.N)
    figure, axes = plt.subplots(len(rows), 6, figsize=(12.5, 2.25 * len(rows)))
    for row_number, row in enumerate(rows):
        wafers = [{"index": row["query_index"], "label": row["query_label"]}, *row["neighbors"]]
        for column, wafer in enumerate(wafers):
            axis = axes[row_number, column]
            axis.imshow(maps[wafer["index"]], cmap=colors, norm=norm, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if column == 0:
                axis.set_title(f"Query: {wafer['label']}", color="#243041", fontsize=13, fontweight="bold")
            else:
                matches = wafer["label"] == row["query_label"]
                axis.set_title(
                    f"{column}. {wafer['label']}",
                    color="#16803d" if matches else "#c62828",
                    fontsize=12,
                    fontweight="bold",
                )
        axes[row_number, 0].set_ylabel(
            f"{row['matching_neighbors']}/5 match",
            rotation=90,
            fontsize=11,
            labelpad=9,
        )
    figure.suptitle("Autoencoder retrieval: fixed test queries and five closest training wafers", fontsize=15)
    figure.text(0.5, 0.012, "Dark = outside wafer     Light = passing die     Red = failing die     Green label = match", ha="center", fontsize=11)
    figure.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.065, hspace=0.45, wspace=0.13)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.figure}")


if __name__ == "__main__":
    main()
