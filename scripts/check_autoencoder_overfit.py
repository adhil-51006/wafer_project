"""Prove that the autoencoder can memorize one small mixed wafer batch."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wafer_retrieval.autoencoder import WaferAutoencoder, one_hot_maps
from wafer_retrieval.evaluation import make_stratified_split
from wafer_retrieval.reproducibility import set_seed
from wafer_retrieval.training import (
    categorical_reconstruction_loss,
    reconstruction_metrics,
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
        "--output",
        type=Path,
        default=project_root / "results" / "m4_overfit_check.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-steps", type=int, default=1_500)
    parser.add_argument("--target-loss", type=float, default=0.02)
    return parser.parse_args()


def select_mixed_batch(labels: np.ndarray, fit_indices: np.ndarray, seed: int) -> np.ndarray:
    """Select seven fit wafers per class plus one additional ``none`` wafer."""
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    fit_labels = labels[fit_indices]
    for label in CLASS_ORDER:
        candidates = fit_indices[fit_labels == label]
        selected.append(rng.choice(candidates, size=7, replace=False))

    used_none = selected[0]
    remaining_none = np.setdiff1d(fit_indices[fit_labels == "none"], used_none)
    selected.append(rng.choice(remaining_none, size=1, replace=False))
    batch_indices = np.concatenate(selected)
    rng.shuffle(batch_indices)
    return batch_indices


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(4)

    maps = np.load(args.maps, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r")
    split = make_stratified_split(labels, seed=args.seed)
    batch_indices = select_mixed_batch(labels, split.fit, args.seed)
    targets = torch.from_numpy(np.asarray(maps[batch_indices]))
    inputs = one_hot_maps(targets)

    model = WaferAutoencoder(embedding_dim=args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    loss_history: list[float] = []
    for step in range(args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = categorical_reconstruction_loss(logits, targets)
        loss_value = float(loss.item())
        loss_history.append(loss_value)

        if step % 100 == 0:
            print(f"step={step:4d} loss={loss_value:.6f}", flush=True)
        if loss_value <= args.target_loss:
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_logits, _ = model(inputs)
        final_loss = float(categorical_reconstruction_loss(final_logits, targets).item())
        metrics = reconstruction_metrics(final_logits, targets)

    result = {
        "seed": args.seed,
        "batch_size": len(batch_indices),
        "classes": {label: int(np.count_nonzero(labels[batch_indices] == label)) for label in CLASS_ORDER},
        "embedding_dim": args.embedding_dim,
        "initial_loss": loss_history[0],
        "final_loss": final_loss,
        "steps": len(loss_history) - 1,
        **metrics,
        "passed": final_loss <= args.target_loss,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(
            f"Failed to reach target loss {args.target_loss} in {args.max_steps} steps"
        )


if __name__ == "__main__":
    main()
