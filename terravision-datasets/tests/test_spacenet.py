import json

import numpy as np
import pytest
from PIL import Image

from terravision_data import audit_split, scene_split
from terravision_data import spacenet as sn
from terravision_data.cli import main
from terravision_data.tiling import TileBox


def square(x, y, s):
    return [[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]]


def write_month(root, aoi, month, ids_to_squares, size=(130, 128), alpha_hole=None, seed=0):
    img_rel, lab_rel = sn.month_files(aoi, month)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[0], size[1], 4), dtype=np.uint8)
    arr[..., 3] = 255
    if alpha_hole:
        arr[alpha_hole] = 0
    (root / aoi / "images_masked").mkdir(parents=True, exist_ok=True)
    (root / aoi / "labels_match_pix").mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGBA").save(root / aoi / img_rel)
    feats = [{"type": "Feature", "properties": {"Id": i},
              "geometry": {"type": "Polygon", "coordinates": [square(*sq)]}}
             for i, sq in ids_to_squares.items()]
    (root / aoi / lab_rel).write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def test_parse_listing_and_month_naming():
    xml = ("<ListBucketResult><Prefix>a/</Prefix><CommonPrefixes><Prefix>a/x/</Prefix>"
           "</CommonPrefixes><Contents><Key>a/f_global_monthly_2018_03_mosaic_X.geojson</Key>"
           "</Contents></ListBucketResult>")
    prefixes, keys = sn.parse_listing(xml)
    assert prefixes == ["a/", "a/x/"] and len(keys) == 1
    img, lab = sn.month_files("AOI", "2018_03")
    assert img == "images_masked/global_monthly_2018_03_mosaic_AOI.tif"
    assert lab.endswith("_Buildings.geojson")
    assert sn.choose_months([f"m{i}" for i in range(24)]) == ["m0", "m12", "m23"]
    with pytest.raises(ValueError):
        sn.choose_months(["a", "b"])


def test_rasterize_area_and_ring_handling():
    mask = sn.rasterize([square(10, 20, 8)], (64, 64))
    assert mask.shape == (64, 64) and mask[24, 14] == 1 and mask[0, 0] == 0
    assert 60 <= mask.sum() <= 100  # about an 8x8 square (outline + fill)
    assert sn.rasterize([[[1, 1], [2, 2]]], (8, 8)).sum() == 0  # degenerate ring ignored


def test_change_mask_marks_only_new_building_ids(tmp_path):
    aoi = "AOI_A"
    write_month(tmp_path, aoi, "2018_01", {1: (10, 10, 12), 2: (60, 60, 12)}, seed=1)
    write_month(tmp_path, aoi, "2019_01", {1: (10, 10, 12), 2: (60, 60, 12), 3: (90, 20, 12)},
                size=(128, 128), seed=2)
    f = lambda m: tuple(tmp_path / aoi / r for r in sn.month_files(aoi, m))  # noqa: E731
    before, after, mask, valid = sn.change_pair(*f("2018_01"), *f("2019_01"))
    # common crop of a 130x128 and a 128x128 image
    assert before.shape == after.shape == (3, 128, 128)
    assert before.dtype == np.float32 and 0 <= before.min() and before.max() <= 1
    assert mask[26, 96] == 1 and mask[16, 16] == 0 and mask[66, 66] == 0  # only id 3 is new
    assert valid.all()


def test_invalid_pixels_are_not_labelled_as_change(tmp_path):
    aoi = "AOI_B"
    write_month(tmp_path, aoi, "m1", {1: (5, 5, 10)}, alpha_hole=(slice(80, 128), slice(0, 128)))
    write_month(tmp_path, aoi, "m2", {1: (5, 5, 10), 2: (90, 90, 15)})
    f = lambda m: tuple(tmp_path / aoi / r for r in sn.month_files(aoi, m))  # noqa: E731
    _, _, mask, valid = sn.change_pair(*f("m1"), *f("m2"))
    assert not valid[100, 50] and mask[95:105, 95:105].sum() == 0  # masked in the earlier date


def test_scene_split_never_shares_a_scene():
    boxes = [TileBox(f"s{i % 7}", r, c, 32) for i in range(7) for r in (0, 32) for c in (0, 32)]
    sp = scene_split(boxes, seed=1)
    scenes = [{boxes[i].scene for i in part} for part in (sp.train, sp.val, sp.test)]
    assert not (scenes[0] & scenes[1] or scenes[0] & scenes[2] or scenes[1] & scenes[2])
    assert audit_split(boxes, sp)["ok"]
    with pytest.raises(ValueError):
        scene_split(boxes[:8])


def test_build_dataset_and_cli_offline(tmp_path, capsys):
    cache = tmp_path / "cache"
    for k in range(6):
        aoi = f"AOI_{k}"
        write_month(cache, aoi, "2018_01", {1: (10, 10, 12)}, seed=k)
        write_month(cache, aoi, "2018_12", {1: (10, 10, 12), 2: (70, 40, 14)}, seed=k + 10)
        write_month(cache, aoi, "2019_12", {1: (10, 10, 12), 2: (70, 40, 14), 3: (30, 90, 10)},
                    seed=k + 20)
    assert sn.local_months(cache, "AOI_0") == ["2018_01", "2018_12", "2019_12"]
    code = main(["spacenet", "--cache", str(cache), "--out", str(tmp_path / "ds"), "--aois", "6",
                 "--tile", "64", "--offline"])
    out = capsys.readouterr().out
    assert code == 0 and "audit ok=True" in out
    from terravision_data import load_dataset
    ds = load_dataset(tmp_path / "ds")
    assert ds.before.shape[1:] == (3, 64, 64) and ds.mask.sum() > 0
    assert len(ds.split.train) > 0 and len(ds.split.test) > 0
