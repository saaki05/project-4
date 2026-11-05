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


def balance_tiles(arrays: tuple, neg_ratio: float | None, rng: np.random.Generator) -> tuple:
    """Keep every tile with change and at most ``neg_ratio`` x as many empty tiles.

    Applied to the training subset only, and identically for every compared method. Most
    tiles contain no change at all, so this removes redundant easy negatives.
    """
    if neg_ratio is None:
        return arrays
    changed = arrays[2].reshape(len(arrays[2]), -1).sum(1) > 0
    empty = np.flatnonzero(~changed)
    keep_empty = rng.permutation(empty)[: int(round(neg_ratio * max(int(changed.sum()), 1)))]
    keep = np.sort(np.concatenate([np.flatnonzero(changed), keep_empty]))
    return tuple(a[keep] for a in arrays)


def _summ(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}


def label_efficiency(ds, fractions=(0.05, 0.1, 0.25, 1.0), seeds=(0, 1, 2),
                     kind="fc-siam-diff", cfg: FitConfig | None = None, ssl_epochs=20,
                     ssl_mode="temporal", widths=WIDTHS, out_path=None, device=None,
                     neg_ratio: float | None = None, val_stop_tiles: int | None = None) -> dict:
    cfg = cfg or FitConfig()
    device = device or pick_device()
    train_idx, val, test = ds.split.train, _arrays(ds, ds.split.val), _arrays(ds, ds.split.test)
    bands = ds.before.shape[1]
    results: dict = {"kind": kind, "fractions": list(fractions), "seeds": list(seeds),
                     "ssl_mode": ssl_mode, "ssl_epochs": ssl_epochs,
                     "n_train_tiles": len(train_idx), "neg_ratio": neg_ratio,
                     "val_stop_tiles": val_stop_tiles, "runs": {}, "summary": {}}
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
            sub = balance_tiles(_arrays(ds, train_idx[order[:n]]), neg_ratio,
                                np.random.default_rng(seed + 1))
            stop_val = val if val_stop_tiles is None else tuple(
                a[np.random.default_rng(0).permutation(len(a))[:val_stop_tiles]] for a in val)
            results["runs"].setdefault("train_tiles_used", {})[f"{frac}@{seed}"] = int(len(sub[0]))
            for init in ("scratch", "ssl"):
                torch.manual_seed(seed)
                model = ChangeNet(kind, bands, widths)
                if init == "ssl":
                    model.encoder.load_state_dict(ssl_state)
                fit(model, sub, stop_val, cfg, seed, device)
                res = evaluate(model, val, test, device)
                results["runs"].setdefault(f"{init}@{frac}", []).append(res)
    for key, runs in results["runs"].items():
        if key in ("ssl_loss", "train_tiles_used"):
            continue
        results["summary"][key] = {m: _summ([r[m] for r in runs]) for m in ("f1", "iou", "kappa")}
    if out_path:
        Path(out_path).write_text(json.dumps(results, indent=2))
    return results


def split_leakage(boxes, before, after, mask, tile, seed=0, kind="fc-siam-diff", cfg=None,
                  widths=WIDTHS, device=None, strategies=("random tiles", "spatial blocks",
                                                          "whole scenes"),
                  neg_ratio: float | None = None, val_stop_tiles: int | None = None) -> dict:
    """Train/evaluate the same model under different ways of splitting the same tiles.

    ``random tiles`` ignores geography; ``spatial blocks`` assigns blocks of a scene to one
    side and purges overlapping tiles; ``whole scenes`` assigns entire scenes. Every split is
    audited for pixel-overlapping tiles across the train/evaluation boundary.
    """
    from terravision_data import purge_overlaps, scene_split, spatial_block_split
    cfg = cfg or FitConfig(epochs=15)
    device = device or pick_device()
    bands = before.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(boxes))
    n_tr, n_va = int(0.6 * len(idx)), int(0.2 * len(idx))
    builders = {
        "random tiles": lambda: TileSplit(idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]),
        "spatial blocks": lambda: purge_overlaps(boxes, spatial_block_split(boxes, 2 * tile,
                                                                           seed=seed)),
        "whole scenes": lambda: scene_split(boxes, seed=seed),
    }
    out = {}
    for name in strategies:
        sp = builders[name]()

        def pick(i):
            return (before[i].astype(np.float32), after[i].astype(np.float32),
                    mask[i].astype(np.float32))

        train = balance_tiles(pick(sp.train), neg_ratio, np.random.default_rng(seed + 1))
        val_full = pick(sp.val)
        stop_val = val_full if val_stop_tiles is None else tuple(
            a[np.random.default_rng(0).permutation(len(a))[:val_stop_tiles]] for a in val_full)
        torch.manual_seed(seed)
        model = ChangeNet(kind, bands, widths)
        fit(model, train, stop_val, cfg, seed, device)
        out[name] = {"audit": audit_split(boxes, sp), "n_train": int(len(sp.train)),
                     "n_test": int(len(sp.test)),
                     "scores": evaluate(model, val_full, pick(sp.test), device)}
    return out


def baselines_report(ds) -> dict:
    train, val, test = (_arrays(ds, s) for s in (ds.split.train, ds.split.val, ds.split.test))
    return {"difference threshold": baselines.difference_baseline(val, test),
            "pixel random forest": baselines.random_forest_baseline(train, test)}
