"""Train any missing seed-specific autoencoders needed by the M5 comparison."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(root=root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.root / ".cache" / "m5"
    checkpoint_dir = args.root / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        checkpoint = checkpoint_dir / f"m5_autoencoder_seed_{seed}.pt"
        if checkpoint.exists() and not args.force:
            print(f"seed={seed}: checkpoint already exists; skipping", flush=True)
            continue
        command = [
            sys.executable,
            str(args.root / "scripts" / "train_autoencoder.py"),
            "--seed",
            str(seed),
            "--checkpoint",
            str(checkpoint),
            "--history",
            str(cache_dir / f"history_{seed}.csv"),
            "--summary",
            str(cache_dir / f"summary_{seed}.json"),
            "--figure",
            str(cache_dir / f"curves_{seed}.png"),
            "--num-threads",
            str(args.num_threads),
            "--max-epochs",
            str(args.max_epochs),
            "--skip-figure",
        ]
        print(f"seed={seed}: training", flush=True)
        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = str(args.root / ".cache" / "matplotlib")
        subprocess.run(command, cwd=args.root, env=environment, check=True)


if __name__ == "__main__":
    main()
