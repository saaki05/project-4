"""Building-change dataset derived from SpaceNet 7 (multi-temporal satellite mosaics).

SpaceNet 7 provides ~24 monthly Planet mosaics (about 4 m per pixel) for 60 training areas of
interest (AOIs) and building footprint polygons per month with persistent building ids.
This module turns it into a *new construction* change-detection task:

    change mask (t0 -> t1) = footprints in month t1 whose building id is absent in month t0

Using ids, not pixel differences of rasterised footprints, avoids thin false changes caused by
annotation jitter between months. The labels are derived here; they are not an official
change-detection benchmark, and annotation gaps or clouds can produce label noise.

Data license: SpaceNet 7 is distributed under CC BY-SA 4.0 (verify at
https://spacenet.ai before redistributing). Imagery and labels are never stored in this
repository. Reference: Van Etten et al., "The Multi-Temporal Urban Development SpaceNet
Dataset", CVPR 2021.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BUCKET = "https://spacenet-dataset.s3.amazonaws.com"
TRAIN_PREFIX = "spacenet/SN7_buildings/train/"
_MONTH = re.compile(r"global_monthly_(\d{4}_\d{2})_mosaic_")


# --------------------------------------------------------------------------- remote access
def parse_listing(xml: str) -> tuple[list[str], list[str]]:
    """Return ``(common_prefixes, keys)`` from an S3 ListObjects XML response."""
    prefixes = re.findall(r"<Prefix>([^<]*)</Prefix>", xml)
    keys = re.findall(r"<Key>([^<]*)</Key>", xml)
    return prefixes, keys


def _list(prefix: str, bucket: str = BUCKET, delimiter: str = "/") -> tuple[list[str], list[str]]:
    q = urllib.parse.urlencode({"prefix": prefix, "delimiter": delimiter, "max-keys": 1000})
    with urllib.request.urlopen(f"{bucket}/?{q}", timeout=60) as r:
        prefixes, keys = parse_listing(r.read().decode())
    return [p for p in prefixes if p != prefix], keys


def list_aois(bucket: str = BUCKET) -> list[str]:
    prefixes, _ = _list(TRAIN_PREFIX, bucket)
    return sorted(p.rstrip("/").split("/")[-1] for p in prefixes)


def list_months(aoi: str, bucket: str = BUCKET) -> list[str]:
    _, keys = _list(f"{TRAIN_PREFIX}{aoi}/labels_match_pix/", bucket)
    return sorted({m.group(1) for k in keys if (m := _MONTH.search(k))})


def fetch(url: str, dest: Path, retries: int = 3) -> Path:
    """Download ``url`` to ``dest`` atomically; existing non-empty files are reused."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for _ in range(retries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as err:  # network errors are retried, then re-raised
            last = err
    raise RuntimeError(f"failed to download {url}") from last


def month_files(aoi: str, month: str) -> tuple[str, str]:
    stem = f"global_monthly_{month}_mosaic_{aoi}"
    return f"images_masked/{stem}.tif", f"labels_match_pix/{stem}_Buildings.geojson"


def download_aoi(aoi: str, months: list[str], cache: Path, bucket: str = BUCKET) -> None:
    for m in months:
        for rel in month_files(aoi, m):
            fetch(f"{bucket}/{TRAIN_PREFIX}{aoi}/{rel}", cache / aoi / rel)


def download(aois: list[str], months_by_aoi: dict[str, list[str]], cache: Path,
             bucket: str = BUCKET, workers: int = 6) -> None:
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda a: download_aoi(a, months_by_aoi[a], cache, bucket), aois))


def choose_months(months: list[str]) -> list[str]:
    """First, middle and last available month (about 12 months apart for a 24-month series)."""
    if len(months) < 3:
        raise ValueError("need at least three months")
    return [months[0], months[len(months) // 2], months[-1]]


# --------------------------------------------------------------------------- loading
def load_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """RGB float32 ``(3, H, W)`` in [0, 1] and a boolean valid-pixel mask ``(H, W)``."""
    arr = np.asarray(Image.open(path).convert("RGBA"))
    rgb = arr[..., :3].astype(np.float32).transpose(2, 0, 1) / 255.0
    return rgb, arr[..., 3] > 0


def load_footprints(path: Path) -> dict[int, list[list[list[float]]]]:
    """Building id -> list of polygon exterior rings in pixel coordinates."""
    out: dict[int, list[list[list[float]]]] = {}
    for feat in json.loads(Path(path).read_text())["features"]:
        geom, bid = feat["geometry"], feat["properties"].get("Id")
        if bid is None or geom is None:
            continue
        if geom["type"] == "Polygon":
            polys = [geom["coordinates"][0]]
        elif geom["type"] == "MultiPolygon":
            polys = [p[0] for p in geom["coordinates"]]
        else:
            continue
        out.setdefault(int(bid), []).extend(polys)
    return out


def rasterize(rings: list[list[list[float]]], shape: tuple[int, int]) -> np.ndarray:
    """Fill polygon rings (x, y pixel coordinates) into a uint8 mask of ``shape = (H, W)``."""
    img = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    for ring in rings:
        pts = [(float(p[0]), float(p[1])) for p in ring]
        if len(pts) >= 3:
            draw.polygon(pts, outline=1, fill=1)
    return np.asarray(img, dtype=np.uint8)


def change_pair(img0: Path, lab0: Path, img1: Path, lab1: Path):
    """Return ``(before, after, new_building_mask, valid)`` cropped to the common extent."""
    b, v0 = load_image(img0)
    a, v1 = load_image(img1)
    h, w = min(b.shape[1], a.shape[1]), min(b.shape[2], a.shape[2])
    b, a, valid = b[:, :h, :w], a[:, :h, :w], v0[:h, :w] & v1[:h, :w]
    old_ids = set(load_footprints(lab0))
    new_rings = [r for bid, rings in load_footprints(lab1).items() if bid not in old_ids
                 for r in rings]
    mask = rasterize(new_rings, (a.shape[1], a.shape[2])) * valid.astype(np.uint8)
    return b, a, mask, valid


# --------------------------------------------------------------------------- tile dataset
def local_months(cache: Path, aoi: str) -> list[str]:
    files = (cache / aoi / "labels_match_pix").glob("*_Buildings.geojson")
    return sorted({m.group(1) for f in files if (m := _MONTH.search(f.name))})


def build_dataset(cache: Path, aois: list[str], months_by_aoi: dict[str, list[str]],
                  tile: int = 64, min_valid: float = 0.9, fractions=(0.6, 0.2, 0.2),
                  seed: int = 0):
    """Tile every consecutive month pair of every AOI and split **by AOI**.

    Tiles whose valid-pixel fraction (in both dates) is below ``min_valid`` are dropped. All
    tiles of an AOI, from every date pair, land in the same split, so no location seen in
    training appears in validation or test.
    """
    from terravision_data.splits import scene_split
    from terravision_data.store import TileDataset
    from terravision_data.tiling import TileBox, tile_origins

    before, after, masks, boxes = [], [], [], []
    for aoi in aois:
        months = months_by_aoi[aoi]
        for m0, m1 in zip(months[:-1], months[1:], strict=True):
            i0, l0 = (cache / aoi / rel for rel in month_files(aoi, m0))
            i1, l1 = (cache / aoi / rel for rel in month_files(aoi, m1))
            b, a, mask, valid = change_pair(i0, l0, i1, l1)
            h, w = mask.shape
            if min(h, w) < tile:
                continue
            for r in tile_origins(h, tile, tile):
                for c in tile_origins(w, tile, tile):
                    if valid[r:r + tile, c:c + tile].mean() < min_valid:
                        continue
                    before.append(b[:, r:r + tile, c:c + tile])
                    after.append(a[:, r:r + tile, c:c + tile])
                    masks.append(mask[r:r + tile, c:c + tile])
                    boxes.append(TileBox(aoi, r, c, tile))
    if not boxes:
        raise ValueError("no valid tiles were produced")
    return TileDataset(np.stack(before), np.stack(after), np.stack(masks).astype(np.uint8), boxes,
                       scene_split(boxes, fractions, seed))
