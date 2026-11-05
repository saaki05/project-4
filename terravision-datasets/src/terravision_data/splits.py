"""Spatially blocked splits and a geometric leakage audit.

Random tile-level splits put neighbouring, overlapping tiles into both train and test, which
inflates scores. Here whole spatial blocks (or whole scenes) are assigned to one side, and
``audit_split`` verifies that no train tile overlaps a test tile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from terravision_data.tiling import TileBox


@dataclass(frozen=True)
class TileSplit:
    train: np.ndarray  # indices into the tile list
    val: np.ndarray
    test: np.ndarray


def spatial_block_split(boxes: list[TileBox], block: int, fractions=(0.6, 0.2, 0.2),
                        seed: int = 0) -> TileSplit:
    """Assign every ``block`` x ``block`` pixel region of every scene to one split.

    A tile belongs to the block containing its centre. Tiles from the same block always
    share a split; to avoid leakage between neighbouring blocks, use ``audit_split`` and
    drop tiles that overlap across splits (see ``purge_overlaps``).
    """
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    keys = [(b.scene, (b.row + b.size // 2) // block, (b.col + b.size // 2) // block)
            for b in boxes]
    uniq = sorted(set(keys))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_tr = max(1, round(fractions[0] * len(uniq)))
    n_va = max(1, round(fractions[1] * len(uniq)))
    which = {}
    for rank, i in enumerate(order):
        which[uniq[i]] = 0 if rank < n_tr else (1 if rank < n_tr + n_va else 2)
    side = np.array([which[k] for k in keys])
    return TileSplit(*(np.flatnonzero(side == s) for s in range(3)))


def _overlap(a: TileBox, b: TileBox) -> bool:
    return (a.scene == b.scene and abs(a.row - b.row) < a.size and abs(a.col - b.col) < a.size)


def overlapping_pairs(boxes: list[TileBox], left: np.ndarray, right: np.ndarray):
    return [(int(i), int(j)) for i in left for j in right if _overlap(boxes[i], boxes[j])]


def audit_split(boxes: list[TileBox], split: TileSplit) -> dict:
    """Count pixel-overlapping tile pairs across split boundaries (must be zero)."""
    report = {
        "train_test": len(overlapping_pairs(boxes, split.train, split.test)),
        "train_val": len(overlapping_pairs(boxes, split.train, split.val)),
        "val_test": len(overlapping_pairs(boxes, split.val, split.test)),
        "shared_indices": int(len(set(split.train) & set(split.test))
                              + len(set(split.train) & set(split.val))
                              + len(set(split.val) & set(split.test))),
    }
    report["ok"] = not any(report[k] for k in report if k != "ok")
    return report


def purge_overlaps(boxes: list[TileBox], split: TileSplit) -> TileSplit:
    """Drop train/val tiles that overlap any evaluation tile of a later split."""
    def keep(indices, against):
        bad = {i for i, _ in overlapping_pairs(boxes, indices, against)}
        return np.array([i for i in indices if int(i) not in bad], dtype=int)

    train = keep(split.train, np.concatenate([split.val, split.test]))
    val = keep(split.val, split.test)
    return TileSplit(train, val, split.test)
