# terravision-models

Change-detection networks and the label-efficiency experiment for the TerraVision study of
**self-supervised pretraining for bi-temporal satellite change detection**.

## Contents

| Module | What it provides |
|---|---|
| `models.py` | U-Net based `ChangeNet` with three fusion strategies from Daudt et al. (ICIP 2018): `fc-ef` (early fusion), `fc-siam-conc`, `fc-siam-diff` |
| `ssl.py` | NT-Xent loss written from scratch, augmentation, and contrastive encoder pretraining with augmentation positives or **temporal** positives (same location, two dates) |
| `train.py` | BCE + Dice loss, early stopping on validation F1, threshold selection on validation only, CUDA mixed precision (used automatically on a GPU) |
| `metrics.py` | precision, recall, F1/Dice, IoU, accuracy, Cohen's kappa on the change class |
| `baselines.py` | thresholded image differencing and a pixel-wise random forest |
| `experiments.py` | `label_efficiency` (scratch vs. SSL init at several label fractions, multi-seed) and `split_leakage` (random tile split vs. spatially blocked split) |

## Research questions

1. With 5%, 10%, 25% and 100% of the labelled training tiles, does SSL pretraining on the
   unlabelled training tiles improve change F1/IoU over random initialisation?
2. How much does a random tile split inflate test scores relative to a spatially blocked
   split of the same data?

## Install and test

```bash
git clone https://github.com/saaki05/project-4 && cd project-4
pip install -e terravision-datasets
cd terravision-models && pip install -e ".[dev]" && pytest
```

## Run

```bash
terravision-data synth --out data/synth            # synthetic data, smoke tests only
terravision-models baselines --data data/synth
terravision-models label-efficiency --data data/synth --out results/le.json \
    --fractions 0.05 0.1 0.25 1.0 --seeds 0 1 2
```

On a machine with a CUDA GPU the same command trains on the GPU with mixed precision.

## Status (honest summary)

- All code paths are unit-tested on synthetic scenes (forward shapes, gradient flow,
  symmetric difference fusion, NT-Xent behaviour, SSL loss decrease, training beats a trivial
  predictor, leakage audit detects a random split, end-to-end CLI).
- **No experiment on real satellite imagery has been run.** The OSCD and LEVIR-CD downloads
  were not reachable from the development container. Synthetic scores demonstrate that the
  pipeline works and say nothing about real change detection.
- A loader for the real OSCD file layout is not included yet; it should be written against the
  downloaded files so it can be tested on real inputs.

## Planned real-data protocol

OSCD (24 Sentinel-2 image pairs; check its license before use): split by scene and spatial
block, purge overlaps with `terravision-datasets`, report mean ± std over at least 3 seeds at
each label fraction, and compare against both baselines.

## License

MIT.
