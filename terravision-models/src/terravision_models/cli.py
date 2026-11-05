"""``terravision-models label-efficiency --data DIR --out results.json``."""

from __future__ import annotations

import argparse
import json

from terravision_data import load_dataset

from terravision_models.experiments import baselines_report, label_efficiency
from terravision_models.train import FitConfig, pick_device


def _train(ds, a) -> None:
    from pathlib import Path

    import numpy as np
    import torch

    from terravision_models.experiments import _arrays, balance_tiles
    from terravision_models.models import ChangeNet
    from terravision_models.train import evaluate, fit

    torch.manual_seed(a.seed)
    train = balance_tiles(_arrays(ds, ds.split.train), a.neg_ratio, np.random.default_rng(a.seed))
    val, test = _arrays(ds, ds.split.val), _arrays(ds, ds.split.test)
    keep = np.random.default_rng(0).permutation(len(val[0]))[: a.val_stop_tiles]
    stop_val = tuple(x[keep] for x in val)  # one shared index keeps before/after/mask aligned
    model = ChangeNet(a.kind, ds.before.shape[1], tuple(a.widths))
    cfg = FitConfig(epochs=a.epochs, lr=a.lr, augment=a.augment, cosine=a.cosine,
                    patience=a.patience)
    info = fit(model, train, stop_val, cfg, a.seed)
    res = evaluate(model, val, test)
    out = {"kind": a.kind, "widths": a.widths, "epochs_run": len(info["history"]),
           "best_val_f1": info["best_val_f1"], "test": res, "train_tiles": int(len(train[0])),
           "config": {"lr": a.lr, "augment": a.augment, "cosine": a.cosine, "seed": a.seed,
                      "neg_ratio": a.neg_ratio}, "history": info["history"]}
    Path(a.out).write_text(json.dumps(out, indent=2))
    torch.save({"format_version": 1, "kind": a.kind, "bands": int(ds.before.shape[1]),
                "widths": list(a.widths), "threshold": res["threshold"],
                "tile": int(ds.before.shape[-1]), "state_dict": model.state_dict()},
               a.checkpoint)
    print(f"val F1 {info['best_val_f1']:.3f} | test F1 {res['f1']:.3f} IoU {res['iou']:.3f} "
          f"P {res['precision']:.3f} R {res['recall']:.3f} (threshold {res['threshold']:.2f})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="terravision-models")
    sub = ap.add_subparsers(dest="command", required=True)
    le = sub.add_parser("label-efficiency")
    le.add_argument("--data", required=True, help="dataset directory from terravision-data")
    le.add_argument("--out", required=True)
    le.add_argument("--kind", default="fc-siam-diff", choices=["fc-siam-conc", "fc-siam-diff"])
    le.add_argument("--fractions", nargs="+", type=float, default=[0.05, 0.1, 0.25, 1.0])
    le.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    le.add_argument("--epochs", type=int, default=30)
    le.add_argument("--ssl-epochs", type=int, default=20)
    le.add_argument("--ssl-mode", default="temporal", choices=["augment", "temporal"])
    le.add_argument("--neg-ratio", type=float, default=None,
                    help="keep at most this many empty tiles per tile with change (training only)")
    le.add_argument("--val-stop-tiles", type=int, default=None,
                    help="use a random validation subset of this size for early stopping")
    lk = sub.add_parser("leakage", help="same model under random / blocked / whole-scene splits")
    lk.add_argument("--data", required=True)
    lk.add_argument("--out", required=True)
    lk.add_argument("--seeds", nargs="+", type=int, default=[0])
    lk.add_argument("--epochs", type=int, default=10)
    lk.add_argument("--neg-ratio", type=float, default=None)
    lk.add_argument("--val-stop-tiles", type=int, default=None)
    tr = sub.add_parser("train", help="train one configuration and save a checkpoint")
    tr.add_argument("--data", required=True)
    tr.add_argument("--out", required=True, help="results JSON")
    tr.add_argument("--checkpoint", required=True, help="where to save the model")
    tr.add_argument("--kind", default="fc-siam-diff", choices=["fc-ef", "fc-siam-conc",
                                                              "fc-siam-diff"])
    tr.add_argument("--widths", nargs="+", type=int, default=[16, 32, 64])
    tr.add_argument("--epochs", type=int, default=25)
    tr.add_argument("--lr", type=float, default=2e-3)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--neg-ratio", type=float, default=1.0)
    tr.add_argument("--val-stop-tiles", type=int, default=1200)
    tr.add_argument("--augment", action="store_true")
    tr.add_argument("--cosine", action="store_true")
    tr.add_argument("--patience", type=int, default=8)
    bl = sub.add_parser("baselines")
    bl.add_argument("--data", required=True)
    a = ap.parse_args(argv)

    ds = load_dataset(a.data)
    if a.command == "baselines":
        print(json.dumps(baselines_report(ds), indent=2))
        return 0
    print(f"device: {pick_device()}")
    if a.command == "train":
        _train(ds, a)
        return 0
    if a.command == "leakage":
        from pathlib import Path

        from terravision_models.experiments import split_leakage

        size = ds.boxes[0].size
        res = {str(seed): split_leakage(ds.boxes, ds.before, ds.after, ds.mask, size, seed,
                                        cfg=FitConfig(epochs=a.epochs), neg_ratio=a.neg_ratio,
                                        val_stop_tiles=a.val_stop_tiles) for seed in a.seeds}
        Path(a.out).write_text(json.dumps(res, indent=2, default=int))
        for seed, strategies in res.items():
            for name, r in strategies.items():
                sc = r["scores"]
                print(f"seed {seed} {name:<15} F1 {sc['f1']:.3f}  IoU {sc['iou']:.3f}  "
                      f"overlapping train/test pairs {r['audit']['train_test']}")
        return 0
    res = label_efficiency(ds, tuple(a.fractions), tuple(a.seeds), a.kind,
                           FitConfig(epochs=a.epochs), a.ssl_epochs, a.ssl_mode, out_path=a.out,
                           neg_ratio=a.neg_ratio, val_stop_tiles=a.val_stop_tiles)
    for key, s in res["summary"].items():
        print(f"{key:<14} F1 {s['f1']['mean']:.3f} ± {s['f1']['std']:.3f}   "
              f"IoU {s['iou']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
