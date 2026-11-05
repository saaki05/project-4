"""Generate every TerraVision figure.

Usage: python scripts/make_figures.py [--tiles DIR --cache DIR --checkpoint FILE]
Result-based figures need only the JSON files; the dataset overview needs the tile dataset
(``--tiles``) and the scene overlay needs the tiles, the SpaceNet cache and a checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "terravision-models" / "results" / "spacenet"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
fs.setup()


def label_efficiency():
    d = json.load(open(RES / "label_efficiency.json"))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fracs = d["fractions"]
    for init, col, off in (("scratch", fs.GRAY, -0.012), ("ssl", fs.BLUE, 0.012)):
        means, stds = [], []
        for f in fracs:
            vals = np.array([r["f1"] for r in d["runs"][f"{init}@{f}"]])
            ax.scatter([f * (1 + 6 * off)] * len(vals), vals, color=col, alpha=0.35, s=22, zorder=2)
            means.append(vals.mean())
            stds.append(vals.std())
        xs = [f * (1 + 6 * off) for f in fracs]
        ax.errorbar(xs, means, yerr=stds, color=col, lw=2.2, marker="o", ms=6, capsize=3,
                    label="self-supervised pretraining" if init == "ssl" else "random initialisation",
                    zorder=3, markeredgecolor=fs.SURFACE)
    ax.set_xscale("log")
    ax.set_xticks(fracs, [f"{f:.0%}" for f in fracs])
    ax.minorticks_off()
    ax.set_xlabel("share of labelled training tiles used")
    ax.set_ylabel("test F1 on the change class")
    ax.set_title("Test F1 against the share of labelled tiles used (3 seeds)")
    ax.legend(loc="lower right")
    fs.save(fig, OUT / "label_efficiency.png")


def leakage():
    d = json.load(open(RES / "leakage.json"))
    strategies = ["random tiles", "spatial blocks", "whole scenes"]
    names = {"random tiles": "random tiles", "spatial blocks": "spatial blocks\n(overlaps purged)",
             "whole scenes": "whole areas"}
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    cols = {"random tiles": fs.ORANGE, "spatial blocks": fs.GRAY, "whole scenes": fs.BLUE}
    for i, s in enumerate(strategies):
        vals = [d[seed][s]["scores"]["f1"] for seed in d]
        ov = [d[seed][s]["audit"]["train_test"] for seed in d]
        ax.bar(i, np.mean(vals), 0.55, color=cols[s])
        ax.scatter([i] * len(vals), vals, color=fs.INK, s=22, zorder=3)
        ax.text(i, max(vals) + 0.014, f"mean {np.mean(vals):.3f}", ha="center", color=fs.INK2,
                fontsize=9)
        ax.text(i, 0.012, f"{int(np.mean(ov)):,} overlapping\ntrain/test tile pairs", ha="center",
                fontsize=8, color="white" if s != "spatial blocks" else fs.INK)
    ax.set_xticks(range(3), [names[s] for s in strategies])
    ax.set_ylabel("test F1 (dots = the two seeds)")
    ax.set_ylim(0, 0.36)
    ax.set_title("Random tile splits score higher than whole-area splits (2 seeds)")
    ax.grid(axis="x", visible=False)
    fs.save(fig, OUT / "split_leakage.png")


def baselines_vs_deep():
    b = json.load(open(RES / "baselines.json"))
    rows = [("image difference\n+ threshold", b["difference threshold"]["f1"], fs.GRAY),
            ("pixel random\nforest", b["pixel random forest"]["f1"], fs.GRAY)]
    le = json.load(open(RES / "label_efficiency.json"))
    rows.append(("Siamese U-Net,\n15 epochs, 3 seeds", np.mean([r["f1"] for r in le["runs"]["scratch@1.0"]]),
                 fs.BLUE))
    best = json.load(open(RES / "config_siam_conc_64px.json"))
    rows.append(("best configuration\n(concatenation fusion,\n25 epochs, augment)",
                 best["test"]["f1"], fs.BLUE))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, (n, v, c) in enumerate(rows):
        ax.bar(i, v, 0.55, color=c)
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", color=fs.INK2, fontsize=9)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.set_ylabel("test F1 on the change class")
    ax.set_title("Test F1: pixel baselines against Siamese U-Nets")
    ax.grid(axis="x", visible=False)
    fs.save(fig, OUT / "baselines_vs_deep.png")


def configurations():
    names = {"config_siam_diff_64px": "Siamese, |difference|\n64 px tiles",
             "config_siam_diff_128px": "Siamese, |difference|\n128 px tiles",
             "config_early_fusion_64px": "early fusion\n64 px tiles",
             "config_siam_conc_64px": "Siamese, concatenation\n64 px tiles"}
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i, (stem, label) in enumerate(names.items()):
        d = json.load(open(RES / f"{stem}.json"))
        ax.scatter([i - 0.08], [d["best_val_f1"]], color=fs.GRAY, s=70, zorder=3,
                   label="validation F1 (used to choose)" if i == 0 else None)
        ax.scatter([i + 0.08], [d["test"]["f1"]], color=fs.BLUE, s=70, zorder=3,
                   label="test F1 (reported once)" if i == 0 else None)
        ax.text(i + 0.18, d["test"]["f1"], f"{d['test']['f1']:.3f}", va="center", fontsize=8.5,
                color=fs.INK2)
    ax.set_xticks(range(4), list(names.values()), fontsize=8.5)
    ax.set_ylim(0.15, 0.33)
    ax.set_ylabel("F1 on the change class")
    ax.set_title("Four 25-epoch configurations, one seed each")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    fs.save(fig, OUT / "configurations.png")


def dataset_overview(tiles):
    from terravision_data import load_dataset

    ds = load_dataset(tiles)
    idx = [i for i in ds.split.test if ds.mask[i].mean() > 0.05][:5]
    fig, axes = plt.subplots(3, len(idx), figsize=(2.3 * len(idx), 7.2))
    for k, i in enumerate(idx):
        for r, (img, ttl) in enumerate(((ds.before[i], "before"), (ds.after[i], "after"))):
            axes[r, k].imshow(np.clip(img.transpose(1, 2, 0), 0, 1))
            axes[r, k].set_title(ttl, fontsize=9, loc="center", fontweight="normal")
        axes[2, k].imshow(ds.mask[i], cmap=ListedColormap(["#f1f0ec", fs.ORANGE]), vmin=0, vmax=1)
        axes[2, k].set_title("new buildings (label)", fontsize=9, loc="center", fontweight="normal")
        for r in range(3):
            axes[r, k].set_xticks([])
            axes[r, k].set_yticks([])
            axes[r, k].grid(False)
    fig.suptitle("SpaceNet 7-derived change dataset: 64×64 test tiles, ~4 m per pixel", x=0.01,
                 ha="left", fontweight="bold", fontsize=11)
    fs.save(fig, OUT / "dataset_examples.png")


def scene_overlay(tiles, cache, checkpoint):
    from terravision_analytics.inference import detect, load_checkpoint
    from terravision_data import load_dataset
    from terravision_data import spacenet as sn

    ds = load_dataset(tiles)
    scenes = sorted({ds.boxes[i].scene for i in ds.split.test})
    model, meta = load_checkpoint(checkpoint)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
    chosen = []
    for aoi in scenes:
        months = sn.local_months(Path(cache), aoi)
        if len(months) >= 3:
            chosen.append((aoi, months))
    best = []
    for aoi, months in chosen:  # rank test scenes by amount of true change
        i0, l0 = (Path(cache) / aoi / r for r in sn.month_files(aoi, months[0]))
        i1, l1 = (Path(cache) / aoi / r for r in sn.month_files(aoi, months[-1]))
        b, a, mask, valid = sn.change_pair(i0, l0, i1, l1)
        best.append((mask.sum(), aoi, months, b, a, mask))
    best.sort(key=lambda t: -t[0])
    for row, (_, aoi, months, b, a, mask) in enumerate(best[:2]):
        prob = detect(model, b, a, tile=meta["tile"])
        pred = prob >= meta["threshold"]
        rgb = lambda x: np.clip(x.transpose(1, 2, 0), 0, 1)  # noqa: E731
        axes[row, 0].imshow(rgb(b)); axes[row, 0].set_title(f"before ({months[0]})", fontsize=9)
        axes[row, 1].imshow(rgb(a)); axes[row, 1].set_title(f"after ({months[-1]})", fontsize=9)
        grow = lambda m: binary_dilation(m, iterations=2)  # noqa: E731  (display only)
        ov = rgb(a).copy()
        ov[grow(mask > 0)] = 0.35 * ov[grow(mask > 0)] + 0.65 * np.array([0.92, 0.41, 0.2])
        axes[row, 2].imshow(ov); axes[row, 2].set_title("new buildings (label, orange)", fontsize=9)
        tp = pred & (mask > 0)
        fp = pred & (mask == 0)
        fn = (~pred) & (mask > 0)
        vis = np.ones(mask.shape + (3,)) * 0.97
        vis[grow(fp)] = [0.91, 0.48, 0.64]
        vis[grow(fn)] = [0.92, 0.41, 0.2]
        vis[grow(tp)] = [0.16, 0.47, 0.84]
        axes[row, 3].imshow(vis)
        axes[row, 3].set_title("blue hit · pink false alarm · orange missed\n(pixels enlarged for display)", fontsize=9)
        for k in range(4):
            axes[row, k].set_xticks([]); axes[row, k].set_yticks([]); axes[row, k].grid(False)
    fig.suptitle("Held-out areas: new construction between the first and last available month", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fs.save(fig, OUT / "scene_overlay.png")


def architecture():
    fs.pipeline(OUT / "architecture.png", [
        ("datasets", ["SpaceNet 7 months → change masks", "tiling + Hann stitching",
                      "split by whole area", "pixel-overlap leakage audit"]),
        ("models", ["Siamese U-Net (diff / conc)", "contrastive SSL pretraining",
                    "BCE + Dice, augmentation", "label-efficiency + leakage studies"]),
        ("analytics", ["sliding-window scene inference", "polygons → GeoJSON",
                       "change statistics", "CLI + FastAPI"]),
    ], "TerraVision: three packages, one-way dependencies", accent=fs.BLUE, width=10.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles"); ap.add_argument("--cache")
    ap.add_argument("--checkpoint", default=str(RES / "best_model.pt"))
    a = ap.parse_args()
    label_efficiency(); leakage(); baselines_vs_deep(); configurations(); architecture()
    if a.tiles:
        dataset_overview(a.tiles)
    if a.tiles and a.cache and a.checkpoint:
        scene_overlay(a.tiles, a.cache, a.checkpoint)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
