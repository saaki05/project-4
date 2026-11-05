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

## Quick start

```bash
git clone https://github.com/saaki05/project-4 && cd project-4
pip install -e terravision-datasets -e terravision-models -e terravision-analytics
terravision-data synth --out data/synth                      # synthetic, smoke test only
terravision-models label-efficiency --data data/synth --out results/le.json --seeds 0
```

## License

MIT. Satellite datasets carry their own licenses.
