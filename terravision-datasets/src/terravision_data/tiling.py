"""Tiling of large bi-temporal scenes and stitching of per-tile predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileBox:
    scene: str
    row: int  # top-left pixel of the tile inside its scene
    col: int
    size: int


def tile_origins(length: int, size: int, stride: int) -> list[int]:
    """Start offsets covering ``length`` pixels; the last tile is shifted to end at the border."""
    if size > length:
        raise ValueError(f"tile size {size} larger than scene side {length}")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def tile_scene(before: np.ndarray, after: np.ndarray, mask: np.ndarray | None, scene: str,
               size: int, stride: int | None = None):
    """Cut ``(C, H, W)`` image pairs (and an optional ``(H, W)`` mask) into square tiles.

    Returns ``(before_tiles, after_tiles, mask_tiles_or_None, boxes)`` with arrays stacked on
    axis 0.
    """
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("before/after must both be (C, H, W)")
    _, h, w = before.shape
    if mask is not None and mask.shape != (h, w):
        raise ValueError("mask must be (H, W)")
    stride = stride or size
    bt, at, mt, boxes = [], [], [], []
    for r in tile_origins(h, size, stride):
        for c in tile_origins(w, size, stride):
            bt.append(before[:, r:r + size, c:c + size])
            at.append(after[:, r:r + size, c:c + size])
            if mask is not None:
                mt.append(mask[r:r + size, c:c + size])
            boxes.append(TileBox(scene, r, c, size))
    return (np.stack(bt), np.stack(at), np.stack(mt) if mask is not None else None, boxes)


def stitch(tiles: np.ndarray, boxes: list[TileBox], shape: tuple[int, int],
           window: str = "hann") -> np.ndarray:
    """Blend overlapping per-tile outputs ``(N, C, S, S)`` into a ``(C, H, W)`` map.

    A Hann window down-weights tile borders where predictions are least reliable.
    """
    n, c, s, _ = tiles.shape
    if n != len(boxes):
        raise ValueError("tiles and boxes differ in length")
    if window == "hann":
        w1 = np.hanning(s + 2)[1:-1]
        weight = np.outer(w1, w1).astype(np.float64)
    elif window == "uniform":
        weight = np.ones((s, s))
    else:
        raise ValueError("window must be 'hann' or 'uniform'")
    acc = np.zeros((c, *shape))
    norm = np.zeros(shape)
    for t, b in zip(tiles, boxes, strict=True):
        acc[:, b.row:b.row + s, b.col:b.col + s] += t * weight
        norm[b.row:b.row + s, b.col:b.col + s] += weight
    if (norm == 0).any():
        raise ValueError("tiles do not cover the whole scene")
    return acc / norm
