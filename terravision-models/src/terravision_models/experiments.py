"""Label-efficiency and split-leakage experiments.

``label_efficiency`` asks: at 1%, 5%, 10%, 25%, 100% of the labelled training tiles, does
self-supervised encoder pretraining (on the *unlabelled* training tiles) beat random
initialisation? ``split_leakage`` asks how much a random tile split inflates test scores
relative to a spatially blocked split of the same data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from terravision_data import TileSplit, audit_split

from terravision_models import baselines
from terravision_models.models import ChangeNet
from terravision_models.ssl import pretrain
from terravision_models.train import FitConfig, evaluate, fit, pick_device

WIDTHS = (16, 32, 64)


def _arrays(ds, idx):
    b, a, m = ds.subset(idx)
    return b.astype(np.float32), a.astype(np.float32), m.astype(np.float32)


def _summ(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}


def label_efficiency(ds, fractions=(0.05, 0.1, 0.25, 1.0), seeds=(0, 1, 2),
                     kind="fc-siam-diff", cfg: FitConfig | None = None, ssl_epochs=20,
                     ssl_mode="temporal", widths=WIDTHS, out_path=None, device=None) -> dict:
    cfg = cfg or FitConfig()
    device = device or pick_device()
    train_idx, val, test = ds.split.train, _arrays(ds, ds.split.val), _arrays(ds, ds.split.test)
    bands = ds.before.shape[1]
    results: dict = {"kind": kind, "fractions": list(fractions), "seeds": list(seeds),
                     "ssl_mode": ssl_mode, "ssl_epochs": ssl_epochs,
                     "n_train_tiles": len(train_idx), "runs": {}, "summary": {}}
    if kind == "fc-ef":
        raise ValueError("SSL initialisation needs a Siamese variant (shared single-date encoder)")
    for seed in seeds:
        # one SSL encoder per seed, trained on every unlabelled training tile
        ssl_net = ChangeNet(kind, bands, widths)
        tb = torch.from_numpy(ds.before[train_idx].astype(np.float32))
        ta = torch.from_numpy(ds.after[train_idx].astype(np.float32))
        losses = pretrain(ssl_net.encoder, tb, ta, ssl_mode, ssl_epochs, seed=seed, device=device)
        ssl_state = {k: v.clone() for k, v in ssl_net.encoder.state_dict().items()}
        results["runs"].setdefault("ssl_loss", {})[str(seed)] = [losses[0], losses[-1]]
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(train_idx))
        for frac in fractions:
            n = max(2, int(round(frac * len(train_idx))))
            sub = _arrays(ds, train_idx[order[:n]])
            for init in ("scratch", "ssl"):
                torch.manual_seed(seed)
                model = ChangeNet(kind, bands, widths)
                if init == "ssl":
                    model.encoder.load_state_dict(ssl_state)
                fit(model, sub, val, cfg, seed, device)
                res = evaluate(model, val, test, device)
                results["runs"].setdefault(f"{init}@{frac}", []).append(res)
    for key, runs in results["runs"].items():
        if key == "ssl_loss":
            continue
        results["summary"][key] = {m: _summ([r[m] for r in runs]) for m in ("f1", "iou", "kappa")}
    if out_path:
        Path(out_path).write_text(json.dumps(results, indent=2))
    return results


def split_leakage(boxes, before, after, mask, tile, seed=0, kind="fc-ef", cfg=None,
                  widths=WIDTHS, device=None) -> dict:
    """Train/evaluate the same model under a random tile split and a blocked split."""
    from terravision_data import purge_overlaps, spatial_block_split
    cfg = cfg or FitConfig(epochs=15)
    device = device or pick_device()
    bands = before.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(boxes))
    n_tr, n_va = int(0.6 * len(idx)), int(0.2 * len(idx))
    splits = {
        "random tiles": TileSplit(idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]),
        "spatial blocks": purge_overlaps(boxes, spatial_block_split(boxes, 2 * tile, seed=seed)),
    }
    out = {}
    for name, sp in splits.items():
        pick = lambda i: (before[i].astype(np.float32), after[i].astype(np.float32),  # noqa: E731
                          mask[i].astype(np.float32))
        torch.manual_seed(seed)
        model = ChangeNet(kind, bands, widths)
        fit(model, pick(sp.train), pick(sp.val), cfg, seed, device)
        out[name] = {"audit": audit_split(boxes, sp), "n_train": int(len(sp.train)),
                     "n_test": int(len(sp.test)),
                     "scores": evaluate(model, pick(sp.val), pick(sp.test), device)}
    return out


def baselines_report(ds) -> dict:
    train, val, test = (_arrays(ds, s) for s in (ds.split.train, ds.split.val, ds.split.test))
    return {"difference threshold": baselines.difference_baseline(val, test),
            "pixel random forest": baselines.random_forest_baseline(train, test)}
