"""Train the M4 convolutional autoencoder with early stopping on dev loss."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from wafer_retrieval.autoencoder import WaferAutoencoder, one_hot_maps
from wafer_retrieval.evaluation import make_stratified_split
from wafer_retrieval.reproducibility import set_seed
from wafer_retrieval.training import (
    categorical_reconstruction_loss,
    select_autoencoder_training_indices,
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
        "--checkpoint",
        type=Path,
        default=project_root / "checkpoints" / "m4_autoencoder_best.pt",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=project_root / "results" / "m4_training_history.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=project_root / "results" / "m4_training_summary.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=project_root / "figures" / "m4_training_curves.png",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--none-cap", type=int, default=5_000)
    parser.add_argument("--validation-none-cap", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def make_loader(
    maps: torch.Tensor,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(maps),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def run_epoch(
    model: WaferAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_maps = 0
    correct_pixels = 0
    total_pixels = 0
    fail_intersection = 0
    fail_union = 0

    for (targets,) in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits, _ = model(one_hot_maps(targets))
            loss = categorical_reconstruction_loss(logits, targets)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = len(targets)
        total_loss += float(loss.item()) * batch_size
        total_maps += batch_size
        predictions = logits.detach().argmax(dim=1)
        correct_pixels += int(torch.count_nonzero(predictions == targets))
        total_pixels += targets.numel()
        predicted_fail = predictions == 2
        true_fail = targets == 2
        fail_intersection += int(torch.count_nonzero(predicted_fail & true_fail))
        fail_union += int(torch.count_nonzero(predicted_fail | true_fail))

    return {
        "loss": total_loss / total_maps,
        "pixel_accuracy": correct_pixels / total_pixels,
        "fail_iou": fail_intersection / max(fail_union, 1),
    }


def create_training_objects(
    seed: int,
    embedding_dim: int,
    learning_rate: float,
    train_maps: torch.Tensor,
    batch_size: int,
) -> tuple[WaferAutoencoder, torch.optim.Adam, DataLoader]:
    set_seed(seed)
    model = WaferAutoencoder(embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = make_loader(train_maps, batch_size, seed, shuffle=True)
    return model, optimizer, loader


def save_curves(history: list[dict[str, float]], figure_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axes[0].set(title="Categorical reconstruction loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_fail_iou"] for row in history], label="train")
    axes[1].plot(epochs, [row["validation_fail_iou"] for row in history], label="validation")
    axes[1].set(title="Fail-pixel overlap", xlabel="Epoch", ylabel="IoU", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    maps = np.load(args.maps, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r")
    split = make_stratified_split(labels, seed=args.seed)
    train_indices = select_autoencoder_training_indices(
        split.fit, labels, none_cap=args.none_cap, seed=args.seed
    )
    validation_indices = select_autoencoder_training_indices(
        split.dev, labels, none_cap=args.validation_none_cap, seed=args.seed
    )
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("Training and validation rows overlap")

    train_maps = torch.from_numpy(np.asarray(maps[train_indices]))
    validation_maps = torch.from_numpy(np.asarray(maps[validation_indices]))
    validation_loader = make_loader(
        validation_maps, args.batch_size, args.seed, shuffle=False
    )
    print(
        f"Training maps: {len(train_maps):,}; validation maps: {len(validation_maps):,}",
        flush=True,
    )

    # Run one reference epoch, then recreate every random state and confirm that
    # the real epoch 1 produces the identical loss.
    reference_model, reference_optimizer, reference_loader = create_training_objects(
        args.seed,
        args.embedding_dim,
        args.learning_rate,
        train_maps,
        args.batch_size,
    )
    reference_first_loss = run_epoch(
        reference_model, reference_loader, reference_optimizer
    )["loss"]
    del reference_model, reference_optimizer, reference_loader

    model, optimizer, train_loader = create_training_objects(
        args.seed,
        args.embedding_dim,
        args.learning_rate,
        train_maps,
        args.batch_size,
    )
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(1, args.max_epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer)
        validation_metrics = run_epoch(model, validation_loader, optimizer=None)
        if epoch == 1 and train_metrics["loss"] != reference_first_loss:
            raise ValueError(
                "First epoch was not reproducible: "
                f"{reference_first_loss} versus {train_metrics['loss']}"
            )

        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "train_pixel_accuracy": train_metrics["pixel_accuracy"],
            "validation_pixel_accuracy": validation_metrics["pixel_accuracy"],
            "train_fail_iou": train_metrics["fail_iou"],
            "validation_fail_iou": validation_metrics["fail_iou"],
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_fail_iou={validation_metrics['fail_iou']:.4f}",
            flush=True,
        )

        if validation_metrics["loss"] < best_validation_loss - 1e-4:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": args.embedding_dim,
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation_loss": best_validation_loss,
                    "config": vars(args),
                },
                args.checkpoint,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}", flush=True)
                break

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_validation_metrics = run_epoch(model, validation_loader, optimizer=None)

    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=list(history[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(history)
    save_curves(history, args.figure)

    train_counts = Counter(labels[train_indices].tolist())
    validation_counts = Counter(labels[validation_indices].tolist())
    summary = {
        "seed": args.seed,
        "embedding_dim": args.embedding_dim,
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "train_validation_overlap": int(
            np.intersect1d(train_indices, validation_indices).size
        ),
        "train_class_counts": dict(sorted(train_counts.items())),
        "validation_class_counts": dict(sorted(validation_counts.items())),
        "reference_first_epoch_loss": reference_first_loss,
        "actual_first_epoch_loss": history[0]["train_loss"],
        "first_epoch_reproducible": reference_first_loss == history[0]["train_loss"],
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_pixel_accuracy": best_validation_metrics["pixel_accuracy"],
        "best_validation_fail_iou": best_validation_metrics["fail_iou"],
        "elapsed_seconds": time.time() - started,
        "checkpoint": (
            str(args.checkpoint.relative_to(project_root))
            if args.checkpoint.is_relative_to(project_root)
            else str(args.checkpoint)
        ),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
