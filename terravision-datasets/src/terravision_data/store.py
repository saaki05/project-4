"""Tile dataset persistence with a checksum manifest (the contract with terravision-models)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from terravision_data.splits import TileSplit
from terravision_data.tiling import TileBox

SCHEMA_VERSION = 1


@dataclass
class TileDataset:
    before: np.ndarray  # (N, C, S, S) float32
    after: np.ndarray  # (N, C, S, S) float32
    mask: np.ndarray  # (N, S, S) uint8, 1 = change
    boxes: list[TileBox]
    split: TileSplit

    def subset(self, indices: np.ndarray):
        return self.before[indices], self.after[indices], self.mask[indices]


def save_dataset(ds: TileDataset, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "tiles.npz", before=ds.before, after=ds.after, mask=ds.mask,
                        train=ds.split.train, val=ds.split.val, test=ds.split.test)
    meta = [[b.scene, b.row, b.col, b.size] for b in ds.boxes]
    (out / "boxes.json").write_text(json.dumps(meta))
    manifest = {
        "schema_version": SCHEMA_VERSION, "num_tiles": int(ds.before.shape[0]),
        "bands": int(ds.before.shape[1]), "tile_size": int(ds.before.shape[2]),
        "n_train": int(len(ds.split.train)), "n_val": int(len(ds.split.val)),
        "n_test": int(len(ds.split.test)),
        "changed_pixel_fraction": float(ds.mask.mean()),
        "sha256": {f: hashlib.sha256((out / f).read_bytes()).hexdigest()
                   for f in ("tiles.npz", "boxes.json")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_dataset(data_dir: str | Path, verify: bool = True) -> TileDataset:
    d = Path(data_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {manifest['schema_version']}")
    if verify:
        for f, digest in manifest["sha256"].items():
            if hashlib.sha256((d / f).read_bytes()).hexdigest() != digest:
                raise ValueError(f"checksum mismatch for {f}")
    z = np.load(d / "tiles.npz")
    boxes = [TileBox(s, r, c, size) for s, r, c, size in json.loads((d / "boxes.json").read_text())]
    return TileDataset(z["before"], z["after"], z["mask"], boxes,
                       TileSplit(z["train"], z["val"], z["test"]))
