"""Small convolutional autoencoder for categorical 32×32 wafer maps."""

import torch
from torch import nn
from torch.nn import functional as F


def one_hot_maps(maps: torch.Tensor) -> torch.Tensor:
    """Convert integer maps from ``(B, H, W)`` to float ``(B, 3, H, W)``."""
    if maps.ndim != 3:
        raise ValueError(f"Expected maps with shape (B, H, W), got {tuple(maps.shape)}")
    if maps.numel() == 0:
        raise ValueError("Map batch must not be empty")
    if torch.is_floating_point(maps):
        raise TypeError("Categorical map targets must use an integer tensor dtype")
    if not torch.all((maps >= 0) & (maps <= 2)):
        raise ValueError("Categorical maps must contain only 0, 1, and 2")

    # PyTorch one_hot puts categories last: (B, H, W, C). Convolution layers
    # expect channels first, so permute moves C into the second position.
    encoded = F.one_hot(maps.to(torch.int64), num_classes=3)
    return encoded.permute(0, 3, 1, 2).to(torch.float32)


class WaferAutoencoder(nn.Module):
    """Compress one-hot wafer maps to an embedding and reconstruct pixel logits."""

    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive")
        self.embedding_dim = embedding_dim

        # A convolution learns a small spatial pattern and reuses it everywhere
        # in the image. Stride 2 halves height and width at each downsampling step.
        self.encoder_convolutions = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.to_embedding = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

        self.from_embedding = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 4 * 4),
            nn.ReLU(),
        )
        self.decoder_convolutions = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )

    def encode(self, one_hot: torch.Tensor) -> torch.Tensor:
        """Return one embedding vector per input wafer."""
        hidden = self.encoder_convolutions(one_hot)
        return self.to_embedding(hidden.flatten(start_dim=1))

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """Decode embeddings into three unnormalized scores per pixel."""
        hidden = self.from_embedding(embedding).reshape(-1, 128, 4, 4)
        return self.decoder_convolutions(hidden)

    def forward(self, one_hot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reconstruction logits and the bottleneck embedding."""
        if one_hot.ndim != 4 or one_hot.shape[1:] != (3, 32, 32):
            raise ValueError(
                f"Expected one-hot input shape (B, 3, 32, 32), got {tuple(one_hot.shape)}"
            )
        embedding = self.encode(one_hot)
        logits = self.decode(embedding)
        return logits, embedding
