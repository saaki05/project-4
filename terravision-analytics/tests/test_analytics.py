import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from terravision_data import make_scene, tile_scene
from terravision_models.models import ChangeNet
from terravision_models.train import FitConfig, fit

from terravision_analytics.api import create_app
from terravision_analytics.cli import main
from terravision_analytics.inference import detect, load_checkpoint, save_checkpoint
from terravision_analytics.vectorize import change_statistics, components, to_geojson


def test_components_and_statistics():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:5, 2:6] = 1  # 12 px
    mask[10:12, 10:11] = 1  # 2 px (below min_pixels)
    mask[15, 15] = mask[16, 16] = 1  # diagonal neighbours: one component under 8-connectivity
    comps = components(mask, min_pixels=2)
    assert sorted(c["pixels"] for c in comps) == [2, 2, 12]
    stats = change_statistics(mask, pixel_area=4.0, min_pixels=3)
    assert stats["num_regions"] == 1 and stats["changed_area"] == 16 * 4.0
    assert stats["region_area_max"] == 12 * 4.0


def test_geojson_geometry_and_georeferencing():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:4, 3:6] = 1  # rows 2-3, cols 3-5 -> 2 x 3 pixels
    gj = to_geojson(mask, origin=(100.0, 500.0), pixel_size=(10.0, 10.0), min_pixels=1)
    f = gj["features"][0]
    ring = np.array(f["geometry"]["coordinates"][0])
    assert np.allclose(ring[0], ring[-1])  # closed ring
    assert ring[:, 0].min() == 130.0 and ring[:, 0].max() == 160.0  # cols 3..6 -> x
    assert ring[:, 1].max() == 480.0 and ring[:, 1].min() == 460.0  # rows 2..4 -> y decreases
    assert f["properties"]["pixels"] == 6 and f["properties"]["area"] == 600.0
    assert f["properties"]["centroid_x"] == pytest.approx(145.0)
    json.dumps(gj)  # serialisable


def test_single_pixel_line_falls_back_to_a_box():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[3, 1:5] = 1
    gj = to_geojson(mask, min_pixels=1)
    assert len(gj["features"][0]["geometry"]["coordinates"][0]) >= 4


def trained_model(tmp_path):
    parts = [tile_scene(*make_scene(128, 128, bands=3, n_changes=8, seed=i), f"s{i}", 32, 32)
             for i in range(5)]
    b = np.concatenate([p[0] for p in parts])
    a = np.concatenate([p[1] for p in parts])
    m = np.concatenate([p[2] for p in parts]).astype(np.float32)
    n = len(b)
    model = ChangeNet("fc-siam-diff", 3)
    fit(model, (b[:n * 3 // 4], a[:n * 3 // 4], m[:n * 3 // 4]),
        (b[n * 3 // 4:], a[n * 3 // 4:], m[n * 3 // 4:]),
        FitConfig(epochs=10, batch=8, patience=10), 0, torch.device("cpu"))
    path = tmp_path / "model.pt"
    save_checkpoint(model, path, bands=3, widths=(16, 32, 64), threshold=0.5, tile=32)
    return model, path


def test_scene_inference_matches_tile_geometry_and_finds_changes(tmp_path):
    model, path = trained_model(tmp_path)
    loaded, meta = load_checkpoint(path)
    before, after, mask = make_scene(160, 144, bands=3, n_changes=8, seed=99)
    prob = detect(loaded, before, after, tile=32, stride=16, device=torch.device("cpu"))
    assert prob.shape == (160, 144) and 0.0 <= prob.min() and prob.max() <= 1.0
    assert prob[mask == 1].mean() > prob[mask == 0].mean() + 0.2
    other = detect(model.eval(), before, after, tile=32, stride=16, device=torch.device("cpu"))
    assert np.allclose(prob, other, atol=1e-5)  # checkpoint round trip is exact
    assert meta["tile"] == 32


def test_cli_and_api(tmp_path):
    _, path = trained_model(tmp_path)
    before, after, _ = make_scene(96, 96, bands=3, n_changes=5, seed=7)
    np.save(tmp_path / "b.npy", before)
    np.save(tmp_path / "a.npy", after)
    out = tmp_path / "out.geojson"
    assert main(["--checkpoint", str(path), "--before", str(tmp_path / "b.npy"), "--after",
                 str(tmp_path / "a.npy"), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["type"] == "FeatureCollection"

    client = TestClient(create_app(path))
    assert client.get("/health").json()["bands"] == 3
    body = {"before": before.tolist(), "after": after.tolist(), "pixel_size": [10.0, 10.0]}
    r = client.post("/detect", json=body)
    assert r.status_code == 200 and "num_regions" in r.json()["statistics"]
    assert client.post("/detect", json={**body, "before": before[:2].tolist(),
                                        "after": after[:2].tolist()}).status_code == 422
    small = {"before": before[:, :16, :16].tolist(), "after": after[:, :16, :16].tolist()}
    assert client.post("/detect", json=small).status_code == 422
