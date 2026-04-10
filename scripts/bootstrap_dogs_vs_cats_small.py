#!/usr/bin/env python3
"""Create data/kaggle_dogs_vs_cats_small/ for the course notebooks.

Uses TensorFlow Datasets `cats_vs_dogs` (downloads ~786MB on first run) and writes
the same layout as 05A_asirra_the_dogs_vs_cats_dataset.ipynb:

  train/{cat,dog}/     1000 images per class
  validation/{cat,dog}  500 per class
  test/{cat,dog}       1000 per class

Run from anywhere, with repo root as cwd or pass --output-dir.

  python scripts/bootstrap_dogs_vs_cats_small.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.utils import save_img


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Target kaggle_dogs_vs_cats_small directory (default: <repo>/data/kaggle_dogs_vs_cats_small)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_root = args.output_dir or (repo_root / "data" / "kaggle_dogs_vs_cats_small")
    out_root = out_root.resolve()

    quotas = {
        ("train", "cat"): 1000,
        ("train", "dog"): 1000,
        ("validation", "cat"): 500,
        ("validation", "dog"): 500,
        ("test", "cat"): 1000,
        ("test", "dog"): 1000,
    }
    remaining = dict(quotas)
    splits_order = ("train", "validation", "test")

    print("Loading cats_vs_dogs (first run downloads data; can take several minutes)...")
    ds = tfds.load(
        "cats_vs_dogs",
        split="train",
        as_supervised=True,
        shuffle_files=True,
        read_config=tfds.ReadConfig(shuffle_seed=42),
    )

    rng = np.random.default_rng(42)
    seen = 0
    for image, label in ds:
        if sum(remaining.values()) == 0:
            break
        seen += 1
        li = int(label.numpy())
        cls = "cat" if li == 0 else "dog"
        choices = [
            (s, remaining[(s, cls)])
            for s in splits_order
            if remaining[(s, cls)] > 0
        ]
        if not choices:
            continue
        split_names = [c[0] for c in choices]
        weights = np.array([c[1] for c in choices], dtype=np.float64)
        weights /= weights.sum()
        split = str(rng.choice(split_names, p=weights))
        idx = quotas[(split, cls)] - remaining[(split, cls)]
        dest_dir = out_root / split / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{idx:05d}.jpg"
        save_img(dest, image.numpy())
        remaining[(split, cls)] -= 1

    if sum(remaining.values()) != 0:
        print(
            "Could not fill all quotas (dataset exhausted). Remaining:",
            remaining,
            file=sys.stderr,
        )
        return 1

    print(f"Wrote dataset under {out_root} (scanned {seen} source images).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
