"""Helpers that make experiment randomness repeatable."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed each random-number generator used by this project.

    PyTorch uses random numbers when it initializes neural-network weights.
    Fixing its seed lets us reproduce an experiment instead of relying on luck.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Deterministic algorithms trade some speed for repeatable results.
    torch.use_deterministic_algorithms(True)
