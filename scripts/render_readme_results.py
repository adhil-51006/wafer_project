"""Refresh the README score table directly from the generated M5 CSV."""

import csv
from pathlib import Path


CLASS_ORDER = (
    "none", "Center", "Donut", "Edge-Ring", "Edge-Loc", "Loc",
    "Scratch", "Random", "Near-full", "macro_defects",
)
START = "<!-- RESULTS_TABLE_START -->"
END = "<!-- RESULTS_TABLE_END -->"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    readme_path = root / "README.md"
    csv_path = root / "results/m5_comparison.csv"
    with csv_path.open(encoding="utf-8") as source:
        rows = {row["class"]: row for row in csv.DictReader(source)}

    table = [
        "| Query class | Queries per split | Random | Handcrafted | Autoencoder |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in CLASS_ORDER:
        row = rows[label]
        count = "—" if not row["query_count_mean"] else str(int(float(row["query_count_mean"])))
        scores = [
            f"{float(row[f'{method}_mean']):.3f} ± {float(row[f'{method}_std']):.3f}"
            for method in ("random", "baseline", "autoencoder")
        ]
        display_label = "Macro, defects" if label == "macro_defects" else label
        table.append(f"| {display_label} | {count} | {' | '.join(scores)} |")

    readme = readme_path.read_text(encoding="utf-8")
    if readme.count(START) != 1 or readme.count(END) != 1:
        raise ValueError("README result table markers must occur exactly once")
    before, remainder = readme.split(START, 1)
    _, after = remainder.split(END, 1)
    updated = before + START + "\n\n" + "\n".join(table) + "\n\n" + END + after
    readme_path.write_text(updated, encoding="utf-8")
    print(f"Updated {readme_path} from {csv_path.name}")


if __name__ == "__main__":
    main()
