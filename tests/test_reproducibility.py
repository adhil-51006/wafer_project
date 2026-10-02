import random

import numpy as np
import torch

from wafer_retrieval import set_seed


def draw_random_values(seed: int) -> tuple[float, float, float]:
    set_seed(seed)
    return random.random(), float(np.random.random()), float(torch.rand(1).item())


def test_same_seed_reproduces_all_random_draws() -> None:
    assert draw_random_values(42) == draw_random_values(42)


def test_different_seeds_change_random_draws() -> None:
    assert draw_random_values(42) != draw_random_values(43)
