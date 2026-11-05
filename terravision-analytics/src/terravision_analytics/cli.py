"""``terravision-detect --checkpoint m.pt --before a.npy --after b.npy --out changes.geojson``."""

from __future__ import annotations

import argparse
import json

import numpy as np

from terravision_analytics.inference import detect, load_checkpoint
from terravision_analytics.vectorize import change_statistics, to_geojson


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="terravision-detect")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--before", required=True, help=".npy array, shape (C, H, W)")
    ap.add_argument("--after", required=True, help=".npy array, shape (C, H, W)")
    ap.add_argument("--out", required=True, help="GeoJSON output path")
    ap.add_argument("--origin", nargs=2, type=float, default=[0.0, 0.0])
    ap.add_argument("--pixel-size", nargs=2, type=float, default=[1.0, 1.0])
    ap.add_argument("--min-pixels", type=int, default=4)
    a = ap.parse_args(argv)

    model, meta = load_checkpoint(a.checkpoint)
    before, after = np.load(a.before), np.load(a.after)
    prob = detect(model, before, after, tile=meta["tile"])
    mask = prob >= meta["threshold"]
    gj = to_geojson(mask, tuple(a.origin), tuple(a.pixel_size), a.min_pixels, prob)
    with open(a.out, "w") as fh:
        json.dump(gj, fh)
    print(json.dumps(change_statistics(mask, a.pixel_size[0] * a.pixel_size[1], a.min_pixels)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
