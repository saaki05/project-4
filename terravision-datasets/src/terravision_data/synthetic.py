"""Synthetic bi-temporal scenes with known change masks.

For tests and smoke runs only. Backgrounds are smooth random textures; changes are random
ellipses and rectangles whose spectral response differs between dates, plus sensor noise and
a global radiometric shift between dates (a nuisance change that should not be flagged).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def make_scene(height: int = 256, width: int = 256, bands: int = 4, n_changes: int = 6,
               noise: float = 0.03, radiometric_shift: float = 0.1, seed: int = 0):
    """Return ``(before, after, mask)`` with shapes ``(C, H, W)``, ``(C, H, W)``, ``(H, W)``."""
    rng = np.random.default_rng(seed)
    base = np.stack([gaussian_filter(rng.normal(size=(height, width)), 6) for _ in range(bands)])
    base = (base - base.mean()) / (base.std() + 1e-8) * 0.25 + 0.5
    before = base + rng.normal(scale=noise, size=base.shape)
    after = base * (1 + radiometric_shift) + rng.normal(scale=noise, size=base.shape)

    mask = np.zeros((height, width), dtype=np.uint8)
    yy, xx = np.mgrid[:height, :width]
    for _ in range(n_changes):
        cy, cx = rng.integers(20, height - 20), rng.integers(20, width - 20)
        ry, rx = rng.integers(6, 18), rng.integers(6, 18)
        if rng.random() < 0.5:
            shape = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
        else:
            shape = (abs(yy - cy) <= ry) & (abs(xx - cx) <= rx)
        delta = rng.normal(scale=0.4, size=(bands, 1)) + np.sign(rng.normal(size=(bands, 1))) * 0.3
        after[:, shape] += delta
        mask[shape] = 1
    return before.astype(np.float32), np.clip(after, -1, 2).astype(np.float32), mask
