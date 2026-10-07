"""Choose fixed test queries and find their five nearest training wafers."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from sklearn import config_context
from sklearn.neighbors import NearestNeighbors

from wafer_retrieval.evaluation import make_stratified_split
from wafer_retrieval.training import extract_embeddings, load_autoencoder_checkpoint


EXAMPLE_CLASSES = ("none", "Center", "Donut", "Scratch")


def choose_queries(
    test_indices: np.ndarray, labels: np.ndarray, seed: int
) -> np.ndarray:
    """Choose one query per named class before looking at retrieval results."""
    rng = np.random.default_rng(seed)
    chosen = []
    for label in EXAMPLE_CLASSES:
        candidates = test_indices[labels[test_indices] == label]
        if not len(candidates):
            raise ValueError(f"No test query exists for {label}")
        chosen.append(int(rng.choice(candidates)))
    return np.asarray(chosen, dtype=np.int64)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maps", type=Path, default=root / "data/processed/maps_32.npy")
    parser.add_argument("--labels", type=Path, default=root / "data/processed/labels.npy")
    parser.add_argument(
        "--checkpoint", type=Path, default=root / "checkpoints/m5_autoencoder_seed_42.pt"
    )
    parser.add_argument("--manifest", type=Path, default=root / "results/m7_demo.json")
    parser.add_argument("--figure", type=Path, default=root / "figures/m7_retrieval_demo.png")
    args = parser.parse_args()

    torch.set_num_threads(4)
    maps = np.load(args.maps, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r")
    split = make_stratified_split(labels, seed=args.seed)
    database_indices = split.train
    query_indices = choose_queries(split.test, labels, seed=args.seed)
    if np.intersect1d(database_indices, query_indices).size:
        raise ValueError("A demo query appears in the training database")

    model = load_autoencoder_checkpoint(args.checkpoint, expected_seed=args.seed)
    database_vectors = extract_embeddings(model, maps[database_indices])
    query_vectors = extract_embeddings(model, maps[query_indices])
    search = NearestNeighbors(n_neighbors=5, metric="euclidean", algorithm="brute", n_jobs=-1)
    search.fit(database_vectors)
    with config_context(working_memory=128):
        distances, positions = search.kneighbors(query_vectors)

    rows = []
    for query_index, neighbor_positions, neighbor_distances in zip(
        query_indices, positions, distances, strict=True
    ):
        neighbors = database_indices[neighbor_positions]
        query_label = str(labels[query_index])
        rows.append(
            {
                "query_index": int(query_index),
                "query_label": query_label,
                "neighbors": [
                    {
                        "index": int(index),
                        "label": str(labels[index]),
                        "distance": float(distance),
                    }
                    for index, distance in zip(neighbors, neighbor_distances, strict=True)
                ],
                "matching_neighbors": int(np.count_nonzero(labels[neighbors] == query_label)),
            }
        )

    manifest = {
        "seed": args.seed,
        "selection": "one uniformly sampled outer-test query per class, before retrieval",
        "checkpoint": args.checkpoint.name,
        "database_rows": len(database_indices),
        "query_database_overlap": 0,
        "rows": rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Plotting runs separately so Matplotlib does not share PyTorch's CPU runtime.
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("plot_retrieval_demo.py")),
            "--maps", str(args.maps),
            "--manifest", str(args.manifest),
            "--figure", str(args.figure),
        ],
        check=True,
    )
    for row in rows:
        print(f"{row['query_label']}: {row['matching_neighbors']}/5 labels match")


if __name__ == "__main__":
    main()
