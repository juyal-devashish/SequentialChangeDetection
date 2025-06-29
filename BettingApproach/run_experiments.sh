#!/usr/bin/env bash
set -euo pipefail

# Determine project root (assumes this script lives in the project root)
ROOT_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run expt0
echo "===== Running expt0 (Stopping time vs Δ) ====="
python3 "$ROOT_/experiments/expt0.py" \
    --delta_min 0.09 --delta_max 0.25 --num_delta 5 \
    --alpha0 0.05 --num_trials 100 --n_classes 5 \
    --progress_bar --save_figs 

echo "→ expt0 complete; log at results/logs/expt0.log"

# Run expt1
echo "===== Running expt1 (ARL + Delay) ====="
python3 "$ROOT_/experiments/expt1.py" \
    --n_classes 5 \
    --delta0 0.1 --alpha0 0.01 --alpha1 0.01 \
    --num_trials_arl 500 --num_trials_delay 200 \
    --Tc 10 --N_max 5000 \
    --delta1_min 0.1 --delta1_max 0.3 --num_delta1 5 \
    --save_figs

echo "→ expt1 complete; log at results/logs/expt1.log"

echo "All experiments finished. Figures in results/figures, logs in results/logs."
