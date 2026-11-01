#!/usr/bin/env bash
# Multi-round adaptive attacker simulation (reviewer-5 Q3): server averages
# T independent NBFU releases. Effective sigma -> sigma / sqrt(T).
# T in {1, 2, 4, 8} x 2 domains x 1 seed = 8 runs (eps=5, seed 42).
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-1}
ITERS=24000

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run() {
    local domain=$1 query=$2 T=$3 name=$4
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_${domain}.py \
        --geminio-query "$query" \
        --attack hfgradinv \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "mrT${T}" \
        --seed 42 \
        --dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0 \
        --multi-round-T "$T" 2>&1 | tee "$logfile" | tail -2
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

TS=${TS:-"1 2 4 8"}

for T in $TS; do
    run medical pneumonia_descriptive "$T" "medical_mrT${T}_eps5"
    run uav     solar_panels         "$T" "uav_mrT${T}_eps5"
done

echo "Multi-round adaptive attacker sweep complete."
