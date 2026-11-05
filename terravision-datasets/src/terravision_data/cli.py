"""``terravision-data synth --out DIR`` builds a synthetic tile dataset (smoke tests only).

Real scenes: load your before/after arrays and call ``tile_scene`` for each scene, then
``spatial_block_split``, ``purge_overlaps`` and ``save_dataset`` (see the README).
"""

from __future__ import annotations

import argparse

import numpy as np

from terravision_data.splits import audit_split, purge_overlaps, spatial_block_split
from terravision_data.store import TileDataset, save_dataset
from terravision_data.synthetic import make_scene
from terravision_data.tiling import tile_scene


def build_synthetic(n_scenes=6, size=64, scene_side=256, seed=0) -> TileDataset:
    parts = [tile_scene(*make_scene(scene_side, scene_side, seed=seed + i), f"scene{i}", size,
                        stride=size // 2)
             for i in range(n_scenes)]
    boxes = [b for p in parts for b in p[3]]
    split = purge_overlaps(boxes, spatial_block_split(boxes, block=size * 2, seed=seed))
    return TileDataset(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                       np.concatenate([p[2] for p in parts]).astype(np.uint8), boxes, split)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="terravision-data")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--out", required=True)
    s.add_argument("--scenes", type=int, default=6)
    s.add_argument("--tile", type=int, default=64)
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    ds = build_synthetic(a.scenes, a.tile, seed=a.seed)
    report = audit_split(ds.boxes, ds.split)
    manifest = save_dataset(ds, a.out)
    print(f"{manifest['num_tiles']} tiles (train {manifest['n_train']}, val {manifest['n_val']}, "
          f"test {manifest['n_test']}); leakage audit ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
