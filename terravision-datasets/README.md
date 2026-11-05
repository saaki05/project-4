# terravision-datasets

Dataset layer for TerraVision (satellite change detection). Pure NumPy/SciPy, no deep
learning dependency.

## What it does

- **Tiling** of large `(C, H, W)` before/after scenes into square tiles with configurable
  stride; the last row/column of tiles is shifted to end exactly at the scene border.
- **Stitching** of per-tile outputs back to a scene with Hann-window blending of overlaps.
- **Spatially blocked splits** (`spatial_block_split`) that assign whole spatial regions to
  train, validation or test, and `purge_overlaps` to remove tiles that overlap across splits.
- **Leakage audit** (`audit_split`) counting pixel-overlapping tile pairs across splits; a
  random tile split of overlapping tiles fails it, a purged blocked split passes.
- A versioned on-disk format (`tiles.npz`, `boxes.json`, `manifest.json` with SHA-256
  checksums) that `terravision-models` consumes.
- A **synthetic scene generator** with known change masks, a global radiometric shift between
  dates and sensor noise (for tests only).

## Install and test

```bash
git clone https://github.com/saaki05/project-4 && cd project-4/terravision-datasets
pip install -e ".[dev]" && pytest
terravision-data synth --out data/synth
terravision-data spacenet --cache cache/ --out data/sn7 --aois 30 --tile 64   # downloads ~0.5 GB
```

## Using real scenes

```python
from terravision_data import tile_scene, spatial_block_split, purge_overlaps, save_dataset
from terravision_data.store import TileDataset
import numpy as np

parts = [tile_scene(before, after, mask, scene_name, size=64, stride=32)
         for before, after, mask, scene_name in my_scenes]      # arrays you load yourself
boxes = [b for p in parts for b in p[3]]
split = purge_overlaps(boxes, spatial_block_split(boxes, block=128))
save_dataset(TileDataset(np.concatenate([p[0] for p in parts]),
                         np.concatenate([p[1] for p in parts]),
                         np.concatenate([p[2] for p in parts]).astype("uint8"), boxes, split),
             "data/oscd_tiles")
```

## Status

`terravision_data.spacenet` builds a new-construction change dataset from SpaceNet 7 (download,
month-pair loading, id-based change masks, tiling, split by area) and is tested on a fake local
layout. Readers for OSCD / LEVIR-CD are not included: they must be written against the real
files.

## License

MIT. Satellite datasets carry their own licenses.
