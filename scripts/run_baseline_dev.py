"""Extract handcrafted features and evaluate baseline retrieval on the dev split."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np
from sklearn import config_context
from sklearn.neighbors import NearestNeighbors

from wafer_retrieval.evaluation import (
    make_stratified_split,
    precision_at_k,
    sample_queries_by_class,
)
from wafer_retrieval.features import (
    extract_feature_matrix,
    feature_names,
    scale_database_and_queries,
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
        "--maps",
        type=Path,
        default=project_root / "data" / "processed" / "maps_32.npy",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=project_root / "data" / "processed" / "labels.npy",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=project_root / "data" / "processed" / "handcrafted_features.npy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "results" / "m3_baseline_dev.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=project_root / "results" / "m3_baseline_metadata.json",
    )
    parser.add_argument(
        "--feature-names",
        type=Path,
        default=project_root / "results" / "m3_feature_names.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-cap", type=int, default=2_000)
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def load_or_create_features(maps_path: Path, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}", flush=True)
        return np.load(cache_path, mmap_mode="r")

    maps = np.load(maps_path, mmap_mode="r")
    print(f"Extracting {len(maps):,} handcrafted feature rows ...", flush=True)
    started = time.time()
    features = extract_feature_matrix(maps)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    print(f"Feature extraction finished in {time.time() - started:.1f} seconds", flush=True)
    return features


def main() -> None:
    args = parse_args()
    labels = np.load(args.labels, mmap_mode="r")
    features = load_or_create_features(args.maps, args.feature_cache)
    names = feature_names()
    if features.shape != (len(labels), len(names)):
        raise ValueError(
            f"Expected feature shape {(len(labels), len(names))}, got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError("Feature cache contains NaN or infinity")

    split = make_stratified_split(labels, seed=args.seed)
    query_indices = sample_queries_by_class(
        split.dev,
        labels,
        cap_per_class=args.query_cap,
        seed=args.seed,
    )
    if np.intersect1d(split.fit, query_indices).size:
        raise ValueError("Development queries leaked into the fit database")

    database_features, query_features, scaler = scale_database_and_queries(
        features,
        database_indices=split.fit,
        query_indices=query_indices,
    )
    database_labels = np.asarray(labels[split.fit])
    query_labels = np.asarray(labels[query_indices])

    print(
        f"Searching {len(database_labels):,} database rows for "
        f"{len(query_labels):,} dev queries ...",
        flush=True,
    )
    search = NearestNeighbors(
        n_neighbors=args.k,
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1,
    )
    search.fit(database_features)
    with config_context(working_memory=128):
        _, neighbor_positions = search.kneighbors(query_features)
    neighbor_labels = database_labels[neighbor_positions]
    result = precision_at_k(query_labels, neighbor_labels, k=args.k)

    database_counts = Counter(database_labels.tolist())
    rows: list[dict[str, object]] = []
    for label in CLASS_ORDER:
        prior = database_counts[label] / len(database_labels)
        precision = result.per_class[label]
        rows.append(
            {
                "class": label,
                "database_prior": prior,
                "precision_at_5": precision,
                "lift_over_prior": precision / prior,
                "query_count": result.query_counts[label],
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

    metadata = {
        "seed": args.seed,
        "feature_count": len(names),
        "database_rows": len(split.fit),
        "query_rows": len(query_indices),
        "database_query_overlap": int(np.intersect1d(split.fit, query_indices).size),
        "scaler_fit_rows": len(split.fit),
        "scaler_fit_scope": "fit database only",
        "macro_all": result.macro_all,
        "macro_defects": result.macro_defects,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    args.feature_names.write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")

    print("class          prior    precision@5  lift    queries")
    for row in rows:
        print(
            f"{row['class']:<12} "
            f"{row['database_prior']:>8.4f}  "
            f"{row['precision_at_5']:>11.4f}  "
            f"{row['lift_over_prior']:>6.1f}x  "
            f"{row['query_count']:>7}"
        )
    print(f"Macro, all classes: {result.macro_all:.4f}")
    print(f"Macro, defects only: {result.macro_defects:.4f}")

    for important_class in ("Center", "Edge-Ring"):
        row = next(item for item in rows if item["class"] == important_class)
        if float(row["precision_at_5"]) <= float(row["database_prior"]) + 0.10:
            raise SystemExit(
                f"{important_class} did not beat its prior clearly; inspect neighbors"
            )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
