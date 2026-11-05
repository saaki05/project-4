#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
terravision-models baselines --data tiles64 > results/baselines.json
terravision-models leakage --data tiles64 --out results/leakage.json --seeds 0 1 --epochs 8 --neg-ratio 1 --val-stop-tiles 1200
terravision-models label-efficiency --data tiles64 --out results/label_efficiency.json \
    --fractions 0.05 0.1 0.25 1.0 --seeds 0 1 2 --epochs 10 --ssl-epochs 8 --neg-ratio 1 --val-stop-tiles 1200
