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


def _spacenet(a: argparse.Namespace) -> int:
    from pathlib import Path

    from terravision_data import spacenet as sn

    cache = Path(a.cache)
    if a.offline:
        aois = sorted(p.name for p in cache.iterdir() if p.is_dir())[: a.aois]
        months = {x: sn.local_months(cache, x) for x in aois}
    else:
        all_aois = sn.list_aois()
        pick = np.random.default_rng(a.seed).permutation(len(all_aois))[: a.aois]
        aois = sorted(all_aois[i] for i in pick)
        months = {x: sn.choose_months(sn.list_months(x)) for x in aois}
        sn.download(aois, months, cache)
    months = {x: m for x, m in months.items() if len(m) >= 2}
    ds = sn.build_dataset(cache, sorted(months), months, tile=a.tile, seed=a.seed)
    report = audit_split(ds.boxes, ds.split)
    manifest = save_dataset(ds, a.out)
    print(f"{len(months)} AOIs -> {manifest['num_tiles']} tiles (train {manifest['n_train']}, "
          f"val {manifest['n_val']}, test {manifest['n_test']}); changed-pixel fraction "
          f"{manifest['changed_pixel_fraction']:.4f}; leakage audit ok={report['ok']}")
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="terravision-data")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--out", required=True)
    s.add_argument("--scenes", type=int, default=6)
    s.add_argument("--tile", type=int, default=64)
    s.add_argument("--seed", type=int, default=0)
    sn = sub.add_parser("spacenet", help="build a building-change dataset from SpaceNet 7")
    sn.add_argument("--cache", required=True, help="download cache directory")
    sn.add_argument("--out", required=True)
    sn.add_argument("--aois", type=int, default=30)
    sn.add_argument("--tile", type=int, default=64)
    sn.add_argument("--seed", type=int, default=0)
    sn.add_argument("--offline", action="store_true", help="use only already cached files")
    a = ap.parse_args(argv)
    if a.command == "spacenet":
        return _spacenet(a)
    ds = build_synthetic(a.scenes, a.tile, seed=a.seed)
    report = audit_split(ds.boxes, ds.split)
    manifest = save_dataset(ds, a.out)
    print(f"{manifest['num_tiles']} tiles (train {manifest['n_train']}, val {manifest['n_val']}, "
          f"test {manifest['n_test']}); leakage audit ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
