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
| Experiments on real satellite imagery | **run on a SpaceNet 7-derived new-construction task** (CPU, 3 seeds, 30 areas); OSCD, LEVIR-CD and Sentinel-2 change datasets were unreachable |
| Readers for real data | SpaceNet 7 reader/builder written and tested; OSCD / LEVIR-CD readers **not written** |
| GPU / CUDA / mixed-precision code paths | written, **never executed** (the development container has no GPU); CPU paths are what the tests cover |
| Published numbers | none; see Results above for the measured values and their limits |

The next steps are to run the same pipeline on a standard benchmark such as OSCD (check its
license; a reader must be written against the real files) and to run longer pretraining and
more seeds on a GPU.

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

### Does self-supervised pretraining help with few labels?

`fc-siam-diff` (shared single-date encoder), 10 epochs, 3 seeds, test F1 on the change class
(mean ± std over seeds; per-seed values in the JSON). SSL = contrastive pretraining (8 epochs,
temporal positives: the same tile at both dates, plus augmentation) on the *unlabelled* training
tiles, then supervised fine-tuning. "Label fraction" is the share of the 8,941 training tiles
whose labels are used; empty tiles are then thinned to the same number as tiles with change
(identically for both arms), so the tiles actually trained on were about 210–260, 440–480,
1,100–1,230 and 4,420.

| label fraction | random init F1 | SSL init F1 | SSL − random | seeds where SSL is better |
|---|---|---|---|---|
| 5% | 0.138 ± 0.042 | 0.149 ± 0.064 | +0.011 | 2 / 3 |
| 10% | 0.154 ± 0.008 | 0.170 ± 0.021 | +0.015 | 2 / 3 |
| 25% | 0.187 ± 0.014 | 0.196 ± 0.042 | +0.009 | 2 / 3 |
| 100% | 0.227 ± 0.015 | 0.224 ± 0.015 | −0.002 | 2 / 3 |

What this supports: more labels help clearly (F1 0.14 → 0.23), and the deep model is more than
twice as good as the pixel baselines (0.09–0.11). SSL initialisation is slightly ahead at the
three low label fractions (+0.009 to +0.015) and level at 100%, **but the gaps are no larger than
the seed-to-seed spread with only three seeds**, so this does not establish that SSL helps. The
contrastive loss did fall (3.10 → 1.9) so the pretraining itself ran; with 8 epochs on 8.9k
tiles it is a very short pretraining, and a longer one, more unlabelled data or more seeds
could change the picture.

Other limits: one architecture, 64 px tiles, 10 epochs, CPU only, labels derived from building
footprints (noisy), and 30 areas. Raw values:
[`terravision-models/results/spacenet/label_efficiency.json`](terravision-models/results/spacenet/label_efficiency.json).
Commands: [`terravision-models/results/spacenet/run.sh`](terravision-models/results/spacenet/run.sh).


## Quick start

```bash
git clone https://github.com/saaki05/project-4 && cd project-4
pip install -e terravision-datasets -e terravision-models -e terravision-analytics
terravision-data synth --out data/synth                      # synthetic, smoke test only
terravision-models label-efficiency --data data/synth --out results/le.json --seeds 0
```

## License

MIT. Satellite datasets carry their own licenses.
