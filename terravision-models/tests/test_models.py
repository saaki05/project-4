import numpy as np
import pytest
import torch
from terravision_data import make_scene, tile_scene

from terravision_models import metrics as M
from terravision_models.baselines import difference_baseline, random_forest_baseline
from terravision_models.cli import main
from terravision_models.experiments import label_efficiency, split_leakage
from terravision_models.models import KINDS, ChangeNet
from terravision_models.ssl import augment, nt_xent, pretrain
from terravision_models.train import FitConfig, bce_dice_loss, evaluate, fit

DEV = torch.device("cpu")


@pytest.mark.parametrize("kind", KINDS)
def test_forward_shape_and_gradients(kind):
    net = ChangeNet(kind, in_ch=4)
    b, a = torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32)
    out = net(b, a)
    assert out.shape == (2, 32, 32)
    out.mean().backward()
    assert all(p.grad is not None for p in net.parameters())
    with pytest.raises(ValueError, match="divisible"):
        net(torch.randn(1, 4, 30, 30), torch.randn(1, 4, 30, 30))


def test_siamese_variants_share_encoder_and_diff_is_symmetric():
    net = ChangeNet("fc-siam-diff", 3).eval()
    b, a = torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)
    assert torch.allclose(net(b, a), net(a, b), atol=1e-5)  # |fa - fb| is symmetric
    assert net.encoder.blocks[0][0].in_channels == 3  # one shared single-date encoder
    ef = ChangeNet("fc-ef", 3)
    assert ef.encoder.blocks[0][0].in_channels == 6


def test_metrics_hand_values():
    pred = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    tgt = np.array([1, 0, 1, 0, 0, 0, 0, 0])
    s = M.scores(pred, tgt)
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["f1"] == 0.5
    assert s["iou"] == pytest.approx(1 / 3)
    assert s["accuracy"] == 0.75
    assert M.scores(tgt, tgt)["kappa"] == 1.0
    assert M.scores(np.zeros(8), tgt)["f1"] == 0.0


def test_loss_decreases_when_prediction_improves():
    t = torch.zeros(2, 8, 8)
    t[:, 2:5, 2:5] = 1
    good = (t * 8 - 4)
    bad = torch.zeros_like(t)
    assert bce_dice_loss(good, t) < bce_dice_loss(bad, t)


def test_nt_xent_prefers_matched_pairs():
    torch.manual_seed(0)
    z = torch.randn(16, 8)
    matched = nt_xent(z, z + 0.01 * torch.randn_like(z))
    shuffled = nt_xent(z, z[torch.randperm(16)])
    assert matched < shuffled
    assert matched.item() < torch.log(torch.tensor(31.0)).item()  # better than chance


def test_augment_preserves_shape_and_is_seeded():
    x = torch.randn(4, 3, 16, 16)
    a = augment(x, torch.Generator().manual_seed(1))
    b = augment(x, torch.Generator().manual_seed(1))
    assert a.shape == x.shape and torch.allclose(a, b)


def make_data(n_scenes=4, size=32, seed=0):
    parts = [tile_scene(*make_scene(128, 128, bands=3, n_changes=8, seed=seed + i), f"s{i}", size,
                        stride=size) for i in range(n_scenes)]
    before = np.concatenate([p[0] for p in parts])
    after = np.concatenate([p[1] for p in parts])
    mask = np.concatenate([p[2] for p in parts]).astype(np.float32)
    boxes = [b for p in parts for b in p[3]]
    return before, after, mask, boxes


def test_pretrain_reduces_ssl_loss_for_both_modes():
    before, after, _, _ = make_data()
    tb, ta = torch.from_numpy(before), torch.from_numpy(after)
    for mode in ("temporal", "augment"):
        net = ChangeNet("fc-siam-diff", 3)
        losses = pretrain(net.encoder, tb, ta, mode, epochs=6, batch=16, seed=0)
        assert losses[-1] < losses[0], mode
    with pytest.raises(ValueError):
        pretrain(ChangeNet("fc-siam-diff", 3).encoder, tb, None, "temporal")


def test_training_beats_trivial_and_difference_baseline_works():
    before, after, mask, _ = make_data(n_scenes=6)
    n = len(before)
    tr, va, te = slice(0, n // 2), slice(n // 2, 3 * n // 4), slice(3 * n // 4, n)
    pack = lambda s: (before[s], after[s], mask[s])  # noqa: E731
    torch.manual_seed(0)
    net = ChangeNet("fc-siam-diff", 3)
    info = fit(net, pack(tr), pack(va), FitConfig(epochs=12, batch=8, patience=12), 0, DEV)
    res = evaluate(net, pack(va), pack(te), DEV)
    assert info["best_val_f1"] > 0.3 and res["f1"] > 0.3
    diff = difference_baseline(pack(va), pack(te))
    assert 0.0 <= diff["f1"] <= 1.0
    rf = random_forest_baseline(pack(tr), pack(te), max_pixels=20000)
    assert rf["f1"] > 0.2


def build_ds(tmp_path):
    from terravision_data.cli import build_synthetic
    return build_synthetic(n_scenes=5, size=32, scene_side=128, seed=2)


def test_label_efficiency_end_to_end(tmp_path):
    ds = build_ds(tmp_path)
    out = tmp_path / "le.json"
    res = label_efficiency(ds, fractions=(0.5, 1.0), seeds=(0,), kind="fc-siam-diff",
                           cfg=FitConfig(epochs=3, batch=8, patience=3), ssl_epochs=2,
                           out_path=out, device=DEV)
    assert {"scratch@0.5", "ssl@0.5", "scratch@1.0", "ssl@1.0"} <= set(res["summary"])
    assert out.exists()
    with pytest.raises(ValueError):
        label_efficiency(ds, kind="fc-ef", device=DEV)


def test_split_leakage_reports_both_splits():
    before, after, mask, boxes = make_data(n_scenes=4)
    # overlapping tiles make leakage possible
    parts = [tile_scene(*make_scene(128, 128, bands=3, seed=i), f"s{i}", 32, stride=16)
             for i in range(4)]
    boxes = [b for p in parts for b in p[3]]
    b = np.concatenate([p[0] for p in parts])
    a = np.concatenate([p[1] for p in parts])
    m = np.concatenate([p[2] for p in parts])
    res = split_leakage(boxes, b, a, m, tile=32, cfg=FitConfig(epochs=2, batch=16, patience=2),
                        device=DEV)
    assert res["random tiles"]["audit"]["ok"] is False  # random split leaks
    assert res["spatial blocks"]["audit"]["ok"] is True


def test_cli_label_efficiency(tmp_path, capsys):
    from terravision_data import save_dataset
    save_dataset(build_ds(tmp_path), tmp_path / "d")
    assert main(["baselines", "--data", str(tmp_path / "d")]) == 0
    code = main(["label-efficiency", "--data", str(tmp_path / "d"), "--out",
                 str(tmp_path / "r.json"), "--fractions", "1.0", "--seeds", "0", "--epochs", "2",
                 "--ssl-epochs", "1"])
    assert code == 0 and "scratch@1.0" in capsys.readouterr().out
