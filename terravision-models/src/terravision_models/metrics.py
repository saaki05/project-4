"""Binary segmentation metrics on the change class."""

from __future__ import annotations

import numpy as np


def confusion(pred: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    p, t = pred.astype(bool).ravel(), target.astype(bool).ravel()
    return (int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()), int((~p & ~t).sum()))


def scores(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Precision, recall, F1 (= Dice), IoU, overall accuracy and Cohen's kappa."""
    tp, fp, fn, tn = confusion(pred, target)
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    acc = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "accuracy": acc,
            "kappa": kappa}


def best_threshold(prob: np.ndarray, target: np.ndarray, grid=None) -> float:
    """Probability threshold maximising F1 (select on validation data only)."""
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    return float(max(grid, key=lambda t: scores(prob >= t, target)["f1"]))
