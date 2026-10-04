"""Compare random-retrieval precision@5 with database class priors."""

import argparse
from collections import Counter
import csv
import math
from pathlib import Path

import numpy as np

from wafer_retrieval.evaluation import (
    make_stratified_split,
    precision_at_k,
    random_neighbor_indices,
    sample_queries_by_class,
)


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


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=project_root / "data" / "processed" / "labels.npy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "results" / "m2_random_sanity.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-cap", type=int, default=2_000)
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = np.load(args.labels, mmap_mode="r")
    split = make_stratified_split(labels, seed=args.seed)
    query_indices = sample_queries_by_class(
        split.test,
        labels,
        cap_per_class=args.query_cap,
        seed=args.seed,
    )

    database_labels = np.asarray(labels[split.train])
    query_labels = np.asarray(labels[query_indices])
    random_positions = random_neighbor_indices(
        database_size=len(database_labels),
        query_count=len(query_labels),
        k=args.k,
        seed=args.seed,
    )
    result = precision_at_k(
        query_labels,
        database_labels[random_positions],
        k=args.k,
    )

    database_counts = Counter(database_labels.tolist())
    rows: list[dict[str, object]] = []
    for label in CLASS_ORDER:
        prior = database_counts[label] / len(database_labels)
        query_count = result.query_counts[label]
        standard_error = math.sqrt(prior * (1 - prior) / (query_count * args.k))
        tolerance = 4 * standard_error + 1 / (query_count * args.k)
        measured = result.per_class[label]
        rows.append(
            {
                "class": label,
                "database_prior": prior,
                "measured_precision_at_5": measured,
                "query_count": query_count,
                "tolerance": tolerance,
                "within_tolerance": abs(measured - prior) <= tolerance,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print("class          prior    measured  queries  within tolerance")
    for row in rows:
        print(
            f"{row['class']:<12} "
            f"{row['database_prior']:>8.4f}  "
            f"{row['measured_precision_at_5']:>8.4f}  "
            f"{row['query_count']:>7}  "
            f"{row['within_tolerance']}"
        )

    if not all(bool(row["within_tolerance"]) for row in rows):
        raise SystemExit("Random-retrieval sanity check failed")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
