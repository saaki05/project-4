# TerraVision

Bi-temporal **satellite change detection** with a focus on evaluation honesty and label
efficiency.

Three independently installable packages with one-way dependencies:

```
terravision-datasets  ──▶  terravision-models  ──▶  terravision-analytics
 tiling, stitching,         U-Nets (early fusion,        scene inference, GeoJSON
 blocked splits + audit     Siamese), SSL pretraining,   polygons, statistics, API
 (numpy, scipy)             label-efficiency study (torch)
```

| Package | Responsibility |
|---|---|
| [`terravision-datasets`](terravision-datasets) | tiling / Hann-blended stitching, spatially blocked splits, overlap purge, geometric leakage audit, checksummed tile format, synthetic scenes for tests |
| [`terravision-models`](terravision-models) | `fc-ef` / `fc-siam-conc` / `fc-siam-diff` U-Nets, NT-Xent and contrastive pretraining (augmentation or temporal positives), BCE+Dice training, baselines, experiments |
| [`terravision-analytics`](terravision-analytics) | checkpoints, whole-scene inference, polygon extraction, change statistics, CLI and HTTP API |

## Research questions

1. **Label efficiency.** With 5–100% of the labelled tiles, does self-supervised pretraining
   on the unlabelled before/after tiles improve change F1/IoU over random initialisation?
2. **Evaluation leakage.** How much does a random tile split (overlapping tiles on both
   sides of the boundary) inflate test scores compared with a spatially blocked, purged split?

## Status (honest summary)

| Item | State |
|---|---|
| All three packages implemented and unit-tested on synthetic scenes | yes (shapes, gradients, symmetric fusion, NT-Xent, SSL loss decrease, leakage detection, tile/stitch exactness, GeoJSON georeferencing, API validation) |
| Experiments on real satellite imagery (OSCD, LEVIR-CD, Sentinel-2) | **not run**: the datasets were not reachable from the development container, which also has no GPU |
| Raster/GeoTIFF readers for the real datasets | **not written**: they should be developed against the real files |
| GPU / CUDA / mixed-precision code paths | written, **never executed** (the development container has no GPU); CPU paths are what the tests cover |
| Published numbers | none; synthetic-data scores only show the pipeline works |

The next step is to download OSCD (check its license), write the loader, build the tile
dataset with `terravision-datasets`, and run `terravision-models label-efficiency` on a GPU.

## Results on real satellite imagery (CPU)

### The dataset

Derived from SpaceNet 7 (monthly Planet mosaics, about 4 m per pixel; building footprints
with persistent ids; CC BY-SA 4.0, verify at spacenet.ai): the task is **new construction**.
A pixel is "changed" if it lies in a building footprint present in a later month whose id is
absent in the earlier month. The labels are derived here (they are not an official
change-detection benchmark) and annotation gaps and clouds add label noise. 30 randomly
chosen areas × first / middle / last month give two ~12-month date pairs each; 64 × 64 tiles
with at least 90% valid pixels in both dates: **14,918 tiles**, split **by area**
(18 / 6 / 6 areas → 8,941 / 3,004 / 2,973 tiles). Only 0.53% of pixels are changed; 24.7% of
training tiles contain any change. Build it with
`terravision-data spacenet --cache <dir> --out <dir> --aois 30 --tile 64`.

### Non-deep baselines (test split, change class)

| method | precision | recall | F1 | IoU | kappa |
|---|---|---|---|---|---|
| thresholded image difference | 0.072 | 0.115 | 0.089 | 0.046 | 0.083 |
| pixel random forest | 0.082 | 0.152 | 0.107 | 0.056 | 0.101 |

Simple methods fail because seasonal and illumination differences between months dwarf the
signal of new buildings.

### How much does a leaky split inflate the score? (same model, same data)

`fc-siam-diff`, 8 epochs, empty tiles kept at 1× the number of tiles with change, validation
threshold, **two seeds** (each strategy has a different test set, so the numbers are not
paired). "Overlapping pairs" = train/test tile pairs sharing pixels (the audit):

| split strategy | seed 0 F1 / IoU | seed 1 F1 / IoU | train/test tile pairs overlapping |
|---|---|---|---|
| random tiles | 0.258 / 0.148 | 0.292 / 0.171 | 1,857 / 1,886 |
| spatial blocks (overlaps purged) | 0.225 / 0.127 | 0.295 / 0.173 | 0 / 0 |
| whole areas (scenes) | **0.192 / 0.106** | **0.201 / 0.112** | 0 / 0 |

Random tile splits put the same location (other date pair, adjacent tiles) on both sides; the
strict whole-area split scores lower in both seeds (F1 0.19–0.20 vs 0.26–0.29, a relative
inflation of about 34–45%). Blocked splits sit in between and are noisy (equal to the random
split in seed 1). With only two seeds, treat the size of the gap as indicative. Raw values:
[`terravision-models/results/spacenet/leakage.json`](terravision-models/results/spacenet/leakage.json).

LABEL_EFFICIENCY_SECTION

## Quick start

```bash
git clone https://github.com/saaki05/project-4 && cd project-4
pip install -e terravision-datasets -e terravision-models -e terravision-analytics
terravision-data synth --out data/synth                      # synthetic, smoke test only
terravision-models label-efficiency --data data/synth --out results/le.json --seeds 0
```

## License

MIT. Satellite datasets carry their own licenses.
