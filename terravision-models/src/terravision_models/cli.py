"""``terravision-models label-efficiency --data DIR --out results.json``."""

from __future__ import annotations

import argparse
import json

from terravision_data import load_dataset

from terravision_models.experiments import baselines_report, label_efficiency
from terravision_models.train import FitConfig, pick_device


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
    bl = sub.add_parser("baselines")
    bl.add_argument("--data", required=True)
    a = ap.parse_args(argv)

    ds = load_dataset(a.data)
    if a.command == "baselines":
        print(json.dumps(baselines_report(ds), indent=2))
        return 0
    print(f"device: {pick_device()}")
    res = label_efficiency(ds, tuple(a.fractions), tuple(a.seeds), a.kind,
                           FitConfig(epochs=a.epochs), a.ssl_epochs, a.ssl_mode, out_path=a.out)
    for key, s in res["summary"].items():
        print(f"{key:<14} F1 {s['f1']['mean']:.3f} ± {s['f1']['std']:.3f}   "
              f"IoU {s['iou']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
