"""Run M0's import, version, and repeatability checks."""

import json
import random

import numpy as np
import torch

from wafer_retrieval import set_seed
from wafer_retrieval.versions import collect_versions


def random_draws(seed: int) -> dict[str, list[float]]:
    set_seed(seed)
    return {
        "python": [random.random() for _ in range(3)],
        "numpy": np.random.random(3).tolist(),
        "torch": torch.rand(3).tolist(),
    }


def main() -> None:
    first = random_draws(42)
    second = random_draws(42)
    print("Software versions:")
    print(json.dumps(collect_versions(), indent=2, sort_keys=True))
    print("\nFirst seeded draw:")
    print(json.dumps(first, indent=2))
    print(f"\nRepeated draw is identical: {first == second}")
    if first != second:
        raise SystemExit("Seeded random draws were not reproducible")


if __name__ == "__main__":
    main()
