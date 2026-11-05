"""Non-deep baselines: thresholded image differencing and a pixel-level random forest."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from terravision_models import metrics as M


def _magnitude(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Per-pixel mean absolute spectral difference, shape (N, H, W)."""
    return np.abs(after - before).mean(axis=1)


def difference_baseline(val: tuple, test: tuple) -> dict:
    """Threshold |after - before| at the value that maximises F1 on the validation split."""
    mv, mt = _magnitude(val[0], val[1]), _magnitude(test[0], test[1])
    grid = np.quantile(mv, np.linspace(0.5, 0.995, 60))
    thr = max(grid, key=lambda t: M.scores(mv >= t, val[2])["f1"])
    return {**M.scores(mt >= thr, test[2]), "threshold": float(thr)}


def _pixel_features(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    d = after - before
    feats = np.concatenate([before, after, d, np.abs(d).mean(1, keepdims=True)], axis=1)
    return feats.transpose(0, 2, 3, 1).reshape(-1, feats.shape[1])


def random_forest_baseline(train: tuple, test: tuple, max_pixels: int = 200_000, seed: int = 0):
    """Random forest on per-pixel spectral features (no spatial context)."""
    x, y = _pixel_features(train[0], train[1]), train[2].reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(max_pixels, len(x)), replace=False)
    rf = RandomForestClassifier(100, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
                                random_state=seed).fit(x[idx], y[idx])
    pred = rf.predict(_pixel_features(test[0], test[1])).reshape(test[2].shape)
    return M.scores(pred, test[2])
