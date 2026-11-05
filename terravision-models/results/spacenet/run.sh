#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
terravision-models baselines --data tiles64 > results/baselines.json
terravision-models leakage --data tiles64 --out results/leakage.json --seeds 0 1 --epochs 8 --neg-ratio 1 --val-stop-tiles 1200
terravision-models label-efficiency --data tiles64 --out results/label_efficiency.json \
    --fractions 0.05 0.1 0.25 1.0 --seeds 0 1 2 --epochs 10 --ssl-epochs 8 --neg-ratio 1 --val-stop-tiles 1200

# Configuration comparison (one seed each; the checkpoint with the best validation F1 is kept)
terravision-models train --data tiles64 --kind fc-siam-diff --widths 16 32 64 --epochs 25 --augment --cosine --neg-ratio 1 --val-stop-tiles 1200 --out results/config_siam_diff_64px.json --checkpoint results/model_siam_diff.pt
terravision-models train --data tiles64 --kind fc-ef --widths 16 32 64 --epochs 25 --augment --cosine --neg-ratio 1 --val-stop-tiles 1200 --out results/config_early_fusion_64px.json --checkpoint results/model_ef.pt
terravision-models train --data tiles64 --kind fc-siam-conc --widths 16 32 64 --epochs 25 --augment --cosine --neg-ratio 1 --val-stop-tiles 1200 --out results/config_siam_conc_64px.json --checkpoint results/model_conc.pt
# 128-pixel tiles: terravision-data spacenet --cache cache --out tiles128 --tile 128 --offline
terravision-models train --data tiles128 --kind fc-siam-diff --widths 16 32 64 --epochs 25 --augment --cosine --neg-ratio 1 --val-stop-tiles 300 --out results/config_siam_diff_128px.json --checkpoint results/model_128.pt

# Label efficiency, corrected protocol (best-threshold checkpoint selection, concatenation fusion)
terravision-models label-efficiency --data tiles64 --out results/label_efficiency.json --kind fc-siam-conc \
    --fractions 0.05 0.1 0.25 1.0 --seeds 0 1 2 --epochs 15 --ssl-epochs 10 --augment --cosine \
    --neg-ratio 1 --val-stop-tiles 1200
