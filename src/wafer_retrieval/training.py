"""Training helpers for the categorical wafer-map autoencoder."""

import numpy as np
import torch
from torch.nn import functional as F

from wafer_retrieval.autoencoder import WaferAutoencoder, one_hot_maps


def categorical_reconstruction_loss(
    logits: torch.Tensor,
    target_maps: torch.Tensor,
) -> torch.Tensor:
    """Calculate per-pixel categorical cross-entropy reconstruction loss."""
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError("Logits must have shape (B, 3, H, W)")
    if target_maps.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise ValueError("Target maps must have shape (B, H, W)")
    if torch.is_floating_point(target_maps):
        raise TypeError("Cross-entropy targets must use an integer tensor dtype")

    # Cross-entropy expects raw class scores and one integer category per pixel.
    return F.cross_entropy(logits, target_maps.to(torch.int64))


def reconstruction_metrics(
    logits: torch.Tensor,
    target_maps: torch.Tensor,
) -> dict[str, float]:
    """Return pixel accuracy and fail-pixel intersection-over-union."""
    predictions = logits.argmax(dim=1)
    targets = target_maps.to(torch.int64)
    pixel_accuracy = (predictions == targets).to(torch.float32).mean()

    predicted_fail = predictions == 2
    true_fail = targets == 2
    intersection = torch.count_nonzero(predicted_fail & true_fail)
    union = torch.count_nonzero(predicted_fail | true_fail)
    fail_iou = intersection.to(torch.float32) / union.clamp_min(1)
    return {
        "pixel_accuracy": float(pixel_accuracy.item()),
        "fail_iou": float(fail_iou.item()),
    }


def select_autoencoder_training_indices(
    fit_indices: np.ndarray,
    labels: np.ndarray,
    none_cap: int,
    seed: int,
) -> np.ndarray:
    """Keep all defect fit rows and a reproducible sample of ``none`` rows."""
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    labels = np.asarray(labels)
    if none_cap <= 0:
        raise ValueError("none_cap must be positive")

    fit_labels = labels[fit_indices]
    defect_indices = fit_indices[fit_labels != "none"]
    none_indices = fit_indices[fit_labels == "none"]
    rng = np.random.default_rng(seed)
    if len(none_indices) > none_cap:
        none_indices = rng.choice(none_indices, size=none_cap, replace=False)

    selected = np.concatenate((defect_indices, none_indices))
    rng.shuffle(selected)
    return selected


def extract_embeddings(
    model: WaferAutoencoder,
    wafer_maps: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Return deterministic NumPy embeddings with evaluation mode and no gradients."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(wafer_maps), batch_size):
            maps = torch.from_numpy(np.asarray(wafer_maps[start : start + batch_size]).copy())
            embeddings = model.encode(one_hot_maps(maps))
            batches.append(embeddings.cpu().numpy())

    if not batches:
        return np.empty((0, model.embedding_dim), dtype=np.float32)
    return np.concatenate(batches).astype(np.float32, copy=False)
