#!/usr/bin/env bash
# Verify defense F1 under the true analytic Gaussian calibration (Balle-Wang).
# HF-GradInv on canonical queries, eps in {5, 10}, 3 seeds = 12 cells total.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-1}
ITERS=24000

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run() {
    local domain=$1 query=$2 eps=$3 seed=$4 name=$5
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_${domain}.py \
        --geminio-query "$query" \
        --attack hfgradinv \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "analytic_eps${eps}_s${seed}" \
        --seed "$seed" \
        --dpsgd-epsilon "$eps" --dpsgd-max-grad-norm 1.0 2>&1 | tee "$logfile" | tail -2
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

SEEDS=${SEEDS:-"42 123 256"}

for SEED in $SEEDS; do
    for EPS in 5 10; do
        run medical pneumonia_descriptive "$EPS" "$SEED" "medical_analytic_eps${EPS}_s${SEED}"
        run uav     solar_panels         "$EPS" "$SEED" "uav_analytic_eps${EPS}_s${SEED}"
    done
done

echo "Analytic Gaussian verification sweep complete."
