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


def scene_split(boxes: list[TileBox], fractions=(0.6, 0.2, 0.2), seed: int = 0) -> TileSplit:
    """Assign whole scenes (e.g. cities) to train / validation / test.

    The strictest generalisation test: no location of a test scene is ever seen in training,
    regardless of acquisition date.
    """
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    scenes = sorted({b.scene for b in boxes})
    if len(scenes) < 3:
        raise ValueError("need at least three scenes")
    order = np.random.default_rng(seed).permutation(len(scenes))
    n_tr = max(1, round(fractions[0] * len(scenes)))
    n_va = max(1, round(fractions[1] * len(scenes)))
    side = {scenes[i]: (0 if r < n_tr else 1 if r < n_tr + n_va else 2)
            for r, i in enumerate(order)}
    which = np.array([side[b.scene] for b in boxes])
    return TileSplit(*(np.flatnonzero(which == k) for k in range(3)))


def _overlap(a: TileBox, b: TileBox) -> bool:
    return (a.scene == b.scene and abs(a.row - b.row) < a.size and abs(a.col - b.col) < a.size)


def overlapping_pairs(boxes: list[TileBox], left: np.ndarray, right: np.ndarray):
    """Index pairs ``(i in left, j in right)`` of tiles that overlap in the same scene."""
    left, right = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
    if len(left) == 0 or len(right) == 0:
        return []
    scenes = np.array([b.scene for b in boxes])
    rows = np.array([b.row for b in boxes])
    cols = np.array([b.col for b in boxes])
    size = np.array([b.size for b in boxes])
    pairs = []
    for scene in set(scenes[left]) & set(scenes[right]):
        li, ri = left[scenes[left] == scene], right[scenes[right] == scene]
        near = ((np.abs(rows[li][:, None] - rows[ri][None, :]) < size[li][:, None])
                & (np.abs(cols[li][:, None] - cols[ri][None, :]) < size[li][:, None]))
        a, b = np.nonzero(near)
        pairs += [(int(li[x]), int(ri[y])) for x, y in zip(a, b, strict=True)]
    return pairs


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
