import numpy as np
import pytest

from terravision_data import (
    TileBox,
    audit_split,
    load_dataset,
    make_scene,
    purge_overlaps,
    save_dataset,
    spatial_block_split,
    stitch,
    tile_origins,
    tile_scene,
)
from terravision_data.cli import build_synthetic, main
from terravision_data.splits import TileSplit


def test_tile_origins_cover_and_end_at_border():
    assert tile_origins(100, 32, 32) == [0, 32, 64, 68]
    assert tile_origins(64, 64, 64) == [0]
    with pytest.raises(ValueError):
        tile_origins(10, 32, 8)


def test_scene_shapes_and_mask():
    before, after, mask = make_scene(128, 160, bands=3, seed=1)
    assert before.shape == after.shape == (3, 128, 160) and mask.shape == (128, 160)
    assert 0.0 < mask.mean() < 0.5
    # changed pixels differ more between dates than unchanged ones
    d = np.abs(after - before).mean(0)
    assert d[mask == 1].mean() > 2 * d[mask == 0].mean()
    b2, _, m2 = make_scene(128, 160, bands=3, seed=1)
    assert np.array_equal(before, b2) and np.array_equal(mask, m2)


def test_tile_then_stitch_recovers_the_scene():
    before, after, mask = make_scene(100, 90, seed=2)
    bt, at, mt, boxes = tile_scene(before, after, mask, "s", size=32, stride=16)
    assert bt.shape[1:] == (4, 32, 32) and mt.shape[1:] == (32, 32)
    rebuilt = stitch(bt, boxes, (100, 90))
    assert np.allclose(rebuilt, before, atol=1e-5)
    uni = stitch(mt[:, None].astype(float), boxes, (100, 90), window="uniform")[0]
    assert np.array_equal(uni.round().astype(int), mask)


def test_stitch_rejects_gaps():
    tiles = np.zeros((1, 1, 8, 8))
    with pytest.raises(ValueError, match="cover"):
        stitch(tiles, [TileBox("s", 0, 0, 8)], (16, 16))


def test_blocked_split_is_leak_free_after_purge():
    parts = [tile_scene(*make_scene(192, 192, seed=i), f"s{i}", 48, stride=24) for i in range(4)]
    boxes = [b for p in parts for b in p[3]]
    split = spatial_block_split(boxes, block=96, seed=0)
    assert set(split.train).isdisjoint(split.test) and len(split.test) > 0
    purged = purge_overlaps(boxes, split)
    assert audit_split(boxes, purged)["ok"]
    assert len(purged.train) <= len(split.train)


def test_audit_detects_random_tile_split_leakage():
    boxes = tile_scene(*make_scene(128, 128, seed=3), "s", 32, stride=16)[3]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(boxes))
    random_split = TileSplit(idx[:20], idx[20:25], idx[25:])
    report = audit_split(boxes, random_split)
    assert not report["ok"] and report["train_test"] > 0


def test_store_round_trip_and_checksum(tmp_path):
    ds = build_synthetic(n_scenes=3, size=32, scene_side=128, seed=1)
    manifest = save_dataset(ds, tmp_path)
    assert manifest["num_tiles"] == len(ds.boxes)
    back = load_dataset(tmp_path)
    assert np.array_equal(back.mask, ds.mask) and back.boxes == ds.boxes
    assert np.array_equal(back.split.test, ds.split.test)
    (tmp_path / "boxes.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(tmp_path)


def test_cli(tmp_path, capsys):
    assert main(["synth", "--out", str(tmp_path), "--scenes", "4", "--tile", "32"]) == 0
    assert "audit ok=True" in capsys.readouterr().out
