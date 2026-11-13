"""Tiny fake MVTec-style dataset for testing the pipeline without the real data.

Not for results – only to check that loading, PatchCore and metrics run end to end
(e.g. on a fresh Colab before the real dataset is copied over).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _texture(rng, size):
    base = rng.normal(0.5, 0.08, (size, size, 1))
    stripes = 0.1 * np.sin(np.arange(size) / 3.0)[None, :, None]
    return np.clip(np.repeat(base + stripes, 3, axis=2), 0, 1)


def _object(rng, size):
    img = np.full((size, size, 3), 0.15) + rng.normal(0, 0.02, (size, size, 3))
    yy, xx = np.mgrid[:size, :size]
    disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size * 0.3) ** 2
    img[disk] = [0.7, 0.6, 0.3]
    return np.clip(img, 0, 1)


def make_synthetic_mvtec(root: str | Path, categories=("carpet", "bottle"), n_train: int = 10,
                         n_test_good: int = 5, n_test_bad: int = 5, size: int = 112,
                         seed: int = 0) -> Path:
    """Write ``root/<cat>/{train,test,ground_truth}`` with square 'scratch' defects."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    for cat in categories:
        gen = _texture if cat in ("carpet", "tile", "wood", "grid", "leather") else _object
        def save(arr, p):
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((arr * 255).astype(np.uint8)).save(p)

        for i in range(n_train):
            save(gen(rng, size), root / cat / "train" / "good" / f"{i:03d}.png")
        for i in range(n_test_good):
            save(gen(rng, size), root / cat / "test" / "good" / f"{i:03d}.png")
        for i in range(n_test_bad):
            img = gen(rng, size)
            mask = np.zeros((size, size))
            s = size // 6
            y, x = rng.integers(s, size - 2 * s, 2)
            img[y:y + s, x:x + s] = [0.95, 0.1, 0.1]
            mask[y:y + s, x:x + s] = 1
            save(img, root / cat / "test" / "scratch" / f"{i:03d}.png")
            mask_dir = root / cat / "ground_truth" / "scratch"
            mask_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_dir / f"{i:03d}_mask.png")
    return root
