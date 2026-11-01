#!/usr/bin/env bash
# Record-level DP-SGD baseline (reviewer-5 Q1): per-example clipping + Gaussian
# noise calibrated to (eps, delta)-DP under substitute-one-record adjacency on
# the per-example average (sensitivity 2C/B).
# 2 domains x 3 seeds = 6 runs at eps=5.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-0}
ITERS=24000

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run() {
    local domain=$1 query=$2 seed=$3 name=$4
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_${domain}.py \
        --geminio-query "$query" \
        --attack hfgradinv \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "reclvl_s${seed}" \
        --seed "$seed" \
        --dpsgd-record-level \
        --dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0 2>&1 | tee "$logfile" | tail -2
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

SEEDS=${SEEDS:-"42 123 256"}

for SEED in $SEEDS; do
    run medical pneumonia_descriptive "$SEED" "medical_reclvl_eps5_s${SEED}"
    run uav     solar_panels         "$SEED" "uav_reclvl_eps5_s${SEED}"
done

echo "Record-level DP-SGD sweep complete."
