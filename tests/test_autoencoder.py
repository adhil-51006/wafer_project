import numpy as np
import pytest
import torch

from wafer_retrieval.autoencoder import WaferAutoencoder, one_hot_maps
from wafer_retrieval.training import (
    categorical_reconstruction_loss,
    extract_embeddings,
    reconstruction_metrics,
)


def test_one_hot_maps_has_three_channels_that_sum_to_one() -> None:
    maps = torch.tensor(
        [
            [[0, 1], [2, 1]],
            [[2, 0], [1, 2]],
        ],
        dtype=torch.uint8,
    )

    encoded = one_hot_maps(maps)

    assert encoded.shape == (2, 3, 2, 2)
    assert encoded.dtype == torch.float32
    torch.testing.assert_close(encoded.sum(dim=1), torch.ones((2, 2, 2)))
    torch.testing.assert_close(encoded[0, :, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(encoded[0, :, 0, 1], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(encoded[0, :, 1, 0], torch.tensor([0.0, 0.0, 1.0]))


def test_autoencoder_output_and_embedding_shapes() -> None:
    maps = torch.randint(0, 3, size=(4, 32, 32), dtype=torch.uint8)
    model = WaferAutoencoder(embedding_dim=64)

    logits, embeddings = model(one_hot_maps(maps))

    assert logits.shape == (4, 3, 32, 32)
    assert embeddings.shape == (4, 64)


def test_autoencoder_rejects_wrong_input_shape() -> None:
    model = WaferAutoencoder()

    with pytest.raises(ValueError, match="one-hot input shape"):
        model(torch.zeros((2, 1, 32, 32)))


def test_one_hot_maps_rejects_noncategorical_values() -> None:
    maps = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.uint8)

    with pytest.raises(ValueError, match="only 0, 1, and 2"):
        one_hot_maps(maps)


def test_categorical_loss_and_reconstruction_metrics_use_integer_targets() -> None:
    targets = torch.tensor([[[0, 1], [2, 1]]], dtype=torch.uint8)
    perfect_logits = one_hot_maps(targets) * 20 - 10

    loss = categorical_reconstruction_loss(perfect_logits, targets)
    metrics = reconstruction_metrics(perfect_logits, targets)

    assert loss.item() < 1e-6
    assert metrics == {"pixel_accuracy": 1.0, "fail_iou": 1.0}


def test_embedding_extraction_is_deterministic_numpy_output() -> None:
    maps = torch.randint(0, 3, size=(5, 32, 32), dtype=torch.uint8).numpy()
    model = WaferAutoencoder(embedding_dim=64)

    first = extract_embeddings(model, maps, batch_size=2)
    repeated = extract_embeddings(model, maps, batch_size=2)
    different_batching = extract_embeddings(model, maps, batch_size=3)

    assert isinstance(first, np.ndarray)
    assert first.shape == (5, 64)
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_allclose(first, different_batching, rtol=1e-6, atol=1e-7)
