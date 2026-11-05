"""Checkpoints and sliding-window inference over whole scenes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from terravision_data import stitch, tile_scene
from terravision_models.models import ChangeNet
from terravision_models.train import predict_proba

FORMAT_VERSION = 1


def save_checkpoint(model: ChangeNet, path: str | Path, bands: int, widths: tuple[int, ...],
                    threshold: float, tile: int) -> None:
    torch.save({"format_version": FORMAT_VERSION, "kind": model.kind, "bands": bands,
                "widths": list(widths), "threshold": threshold, "tile": tile,
                "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str | Path) -> tuple[ChangeNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {ck.get('format_version')!r}")
    model = ChangeNet(ck["kind"], ck["bands"], tuple(ck["widths"]))
    model.load_state_dict(ck["state_dict"])
    return model.eval(), {k: ck[k] for k in ("kind", "bands", "threshold", "tile")}


def detect(model: ChangeNet, before: np.ndarray, after: np.ndarray, tile: int = 64,
           stride: int | None = None, batch: int = 16,
           device: torch.device | None = None) -> np.ndarray:
    """Probability of change for every pixel of a ``(C, H, W)`` before/after pair.

    Overlapping tile predictions are blended with a Hann window, which suppresses seams.
    """
    stride = stride or tile // 2
    bt, at, _, boxes = tile_scene(before.astype(np.float32), after.astype(np.float32), None,
                                  "scene", tile, stride)
    prob = predict_proba(model, bt, at, batch, device)
    return stitch(prob[:, None], boxes, before.shape[1:])[0]
