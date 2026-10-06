"""Compare random, handcrafted, and autoencoder retrieval over repeated splits."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from sklearn import config_context
from sklearn.neighbors import NearestNeighbors

from wafer_retrieval.autoencoder import WaferAutoencoder
from wafer_retrieval.evaluation import (
    make_stratified_split,
    precision_at_k,
    random_neighbor_indices,
    retrieval_confusion,
    sample_queries_by_class,
)
from wafer_retrieval.features import scale_database_and_queries
from wafer_retrieval.training import extract_embeddings


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
    parser.add_argument("--maps", type=Path, default=root / "data/processed/maps_32.npy")
    parser.add_argument("--labels", type=Path, default=root / "data/processed/labels.npy")
    parser.add_argument(
        "--features",
        type=Path,
        default=root / "data/processed/handcrafted_features.npy",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=root / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    parser.add_argument("--figure", type=Path, default=root / "figures/m5_comparison.png")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--query-cap", type=int, default=2_000)
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def checkpoint_for_seed(directory: Path, seed: int) -> Path:
    candidate = directory / f"m5_autoencoder_seed_{seed}.pt"
    if candidate.exists():
        return candidate
    if seed == 42:
        m4_checkpoint = directory / "m4_autoencoder_best.pt"
        if m4_checkpoint.exists():
            return m4_checkpoint
    raise FileNotFoundError(
        f"Missing autoencoder checkpoint for seed {seed}: expected {candidate}"
    )


def index_digest(indices: np.ndarray) -> str:
    """Make a short audit fingerprint for an exact set of row indices."""
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()[:16]


def nearest_labels(
    database_vectors: np.ndarray,
    query_vectors: np.ndarray,
    database_labels: np.ndarray,
    k: int,
) -> np.ndarray:
    """Run exact Euclidean k-nearest-neighbor search and return its labels."""
    search = NearestNeighbors(
        n_neighbors=k, metric="euclidean", algorithm="brute", n_jobs=-1
    )
    search.fit(database_vectors)
    with config_context(working_memory=128):
        _, positions = search.kneighbors(query_vectors)
    return database_labels[positions]


def load_model(checkpoint_path: Path, expected_seed: int) -> WaferAutoencoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["seed"] != expected_seed:
        raise ValueError(
            f"Checkpoint seed {checkpoint['seed']} does not match split seed {expected_seed}"
        )
    model = WaferAutoencoder(embedding_dim=int(checkpoint["embedding_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def write_confusion_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["query_class", *CLASS_ORDER])
        for label, row in zip(CLASS_ORDER, matrix, strict=True):
            writer.writerow([label, *row.tolist()])


def format_score(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def write_markdown_report(
    path: Path, rows: list[dict[str, object]], seed_count: int
) -> None:
    class_rows = [row for row in rows if row["class"] in CLASS_ORDER]
    lines = [
        "# M5 retrieval comparison",
        "",
        f"Scores are mean ± sample standard deviation across {seed_count} stratified splits. "
        "All methods used the same database and queries within each seed.",
        "",
        "| Class | Queries/seed | Random | Baseline | Autoencoder | Note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        count = "—" if row["query_count_mean"] == "" else f"{float(row['query_count_mean']):.1f}"
        note = "few queries; noisy" if row["noisy"] else ""
        lines.append(
            f"| {row['class']} | {count} | "
            f"{format_score(float(row['random_mean']), float(row['random_std']))} | "
            f"{format_score(float(row['baseline_mean']), float(row['baseline_std']))} | "
            f"{format_score(float(row['autoencoder_mean']), float(row['autoencoder_std']))} | {note} |"
        )

    baseline_wins = [
        str(row["class"])
        for row in class_rows
        if float(row["baseline_mean"]) > float(row["autoencoder_mean"])
    ]
    autoencoder_wins = [
        str(row["class"])
        for row in class_rows
        if float(row["autoencoder_mean"]) > float(row["baseline_mean"])
    ]
    exceptions = [
        (str(row["class"]), method)
        for row in class_rows
        if row["class"] != "none"
        for method in ("baseline", "autoencoder")
        if float(row[f"{method}_mean"]) <= float(row["random_mean"])
    ]
    lines.extend(
        [
            "",
            f"The baseline has the higher mean on: {', '.join(baseline_wins) or 'no classes'}.",
            f"The autoencoder has the higher mean on: {', '.join(autoencoder_wins) or 'no classes'}.",
            "",
        ]
    )
    if exceptions:
        details = ", ".join(f"{method} on {label}" for label, method in exceptions)
        lines.append(
            f"Methods that did not beat random on a defect class: {details}. "
            "This is reported as a model limitation rather than hidden."
        )
    else:
        lines.append("Both learned and handcrafted retrieval beat random on every defect class.")
    lines.extend(
        [
            "",
            "Near-full is flagged as noisy because each split has fewer than 50 test queries. "
            "The confusion CSV files show the full distribution of retrieved labels for each query label.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    maps = np.load(args.maps, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r")
    features = np.load(args.features, mmap_mode="r")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict[str, object]] = []
    confusions = {method: [] for method in METHODS}
    audit: list[dict[str, object]] = []
    started = time.time()

    for seed in args.seeds:
        split = make_stratified_split(labels, seed=seed)
        database_indices = split.train
        query_indices = sample_queries_by_class(
            split.test, labels, cap_per_class=args.query_cap, seed=seed
        )
        if np.intersect1d(database_indices, query_indices).size:
            raise ValueError("Test queries leaked into the training database")

        database_labels = np.asarray(labels[database_indices])
        query_labels = np.asarray(labels[query_indices])
        print(
            f"seed={seed}: database={len(database_indices):,}, queries={len(query_indices):,}",
            flush=True,
        )

        random_positions = random_neighbor_indices(
            len(database_indices), len(query_indices), args.k, seed
        )
        method_neighbors: dict[str, np.ndarray] = {
            "random": database_labels[random_positions]
        }

        database_features, query_features, _ = scale_database_and_queries(
            features, database_indices, query_indices
        )
        method_neighbors["baseline"] = nearest_labels(
            database_features, query_features, database_labels, args.k
        )
        del database_features, query_features

        checkpoint_path = checkpoint_for_seed(args.checkpoint_dir, seed)
        model = load_model(checkpoint_path, expected_seed=seed)
        # PyTorch produces 64-number vectors; NumPy/scikit-learn performs search.
        database_embeddings = extract_embeddings(model, maps[database_indices])
        query_embeddings = extract_embeddings(model, maps[query_indices])
        method_neighbors["autoencoder"] = nearest_labels(
            database_embeddings, query_embeddings, database_labels, args.k
        )
        del model, database_embeddings, query_embeddings

        # These exact arrays are shared by all three method branches above.
        database_fingerprint = index_digest(database_indices)
        query_fingerprint = index_digest(query_indices)
        audit.append(
            {
                "seed": seed,
                "database_rows": len(database_indices),
                "query_rows": len(query_indices),
                "overlap": 0,
                "database_index_sha256_16": database_fingerprint,
                "query_index_sha256_16": query_fingerprint,
                "checkpoint": checkpoint_path.name,
                "all_methods_used_identical_indices": True,
            }
        )

        for method in METHODS:
            neighbor_labels = method_neighbors[method]
            result = precision_at_k(query_labels, neighbor_labels, k=args.k)
            confusion = retrieval_confusion(query_labels, neighbor_labels, CLASS_ORDER)
            confusions[method].append(confusion)
            for label in CLASS_ORDER:
                per_seed.append(
                    {
                        "seed": seed,
                        "method": method,
                        "class": label,
                        "precision_at_5": result.per_class[label],
                        "query_count": result.query_counts[label],
                    }
                )
            per_seed.extend(
                [
                    {
                        "seed": seed,
                        "method": method,
                        "class": "macro_defects",
                        "precision_at_5": result.macro_defects,
                        "query_count": "",
                    },
                    {
                        "seed": seed,
                        "method": method,
                        "class": "macro_all",
                        "precision_at_5": result.macro_all,
                        "query_count": "",
                    },
                ]
            )
        print(f"seed={seed}: finished all three methods", flush=True)

    per_seed_path = args.output_dir / "m5_per_seed.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(per_seed[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_seed)

    summary_rows: list[dict[str, object]] = []
    for label in (*CLASS_ORDER, "macro_defects", "macro_all"):
        row: dict[str, object] = {"class": label}
        label_records = [record for record in per_seed if record["class"] == label]
        counts = [int(record["query_count"]) for record in label_records if record["query_count"] != ""]
        row["query_count_mean"] = float(np.mean(counts)) if counts else ""
        row["query_count_min"] = min(counts) if counts else ""
        row["query_count_max"] = max(counts) if counts else ""
        row["noisy"] = bool(counts and min(counts) < 50)
        for method in METHODS:
            values = np.array(
                [
                    float(record["precision_at_5"])
                    for record in label_records
                    if record["method"] == method
                ]
            )
            row[f"{method}_mean"] = float(values.mean())
            row[f"{method}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)

    summary_path = args.output_dir / "m5_comparison.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    for method in METHODS:
        mean_confusion = np.mean(np.stack(confusions[method]), axis=0)
        write_confusion_csv(
            args.output_dir / f"m5_confusion_{method}.csv", mean_confusion
        )

    metadata = {
        "seeds": args.seeds,
        "k": args.k,
        "query_cap_per_class": args.query_cap,
        "distance": "euclidean",
        "standard_deviation": "sample standard deviation across seeds (ddof=1)",
        "index_audit": audit,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "m5_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown_report(
        args.output_dir / "m5_report.md", summary_rows, seed_count=len(args.seeds)
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("plot_m5_results.py")),
            "--input",
            str(summary_path),
            "--output",
            str(args.figure),
        ],
        check=True,
    )
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
