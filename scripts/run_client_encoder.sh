#!/usr/bin/env bash
# Client-side robust encoders (prototype/client_encoder.py): the FL client model is a
# CLIP-family ViT-L/14 encoder + Geminio head, so the shared gradient flows through
# the (robust) encoder. Stage "heads" trains one rigged head per (domain, encoder) with
# the attacker VLM held fixed; stage "attack" runs HF-GradInv against each, 3 seeds.
#
# Private batches come from the held-out half of each dataset, so the rigged head never
# saw them (prototype/prepare_target_batch.py): assets/medical_heldout_pna/ (2 pneumonia
# targets of 8) and assets/uav_heldout_solar/ (2 solar-panel targets of 8).
# "resnet18" is the reference: the existing ResNet-18 client with the language-sweep
# heads trained on the same attacker VLM and query (malicious_models_lang_sweep/).
# Runs are resumable: a run with an existing metrics.json is skipped.
#
# Usage: GPUS="3 5 6 7" bash scripts/run_client_encoder.sh [heads|attack|all]
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio

# FARE/TeCoA load through open_clip's Hugging Face hub path; the snapshots are cached
# and hub HEAD requests have timed out on this machine, so stay offline.
export HF_HUB_OFFLINE=1

PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
read -r -a GPU_LIST <<< "${GPUS:-3 5 6 7}"
read -r -a ENCODERS <<< "${ENCODERS:-clip simclip2 simclip4 fare2 fare4 tecoa2 tecoa4 resnet18}"
read -r -a SEEDS <<< "${SEEDS:-42 123 256}"
ITERS=${ITERS:-24000}
STAGE=${1:-all}
LOG_DIR=./results/client_encoder/logs
mkdir -p "$LOG_DIR"

train_heads() {
    local domain=$1 gpu=$2
    for enc in "${ENCODERS[@]}"; do
        [[ $enc == resnet18 ]] && continue
        if [[ -f "malicious_models_client_encoder/${domain}_${enc}.pt" ]]; then
            echo "[$(date +%H:%M:%S)] head ${domain}/${enc} exists, skipping"
            continue
        fi
        echo "[$(date +%H:%M:%S)] head ${domain}/${enc} on GPU ${gpu}"
        "$PY" prototype/client_encoder.py --domain "$domain" --encoder "$enc" --gpu "$gpu" \
            > "$LOG_DIR/head_${domain}_${enc}.log" 2>&1 \
            || echo "[$(date +%H:%M:%S)] FAILED head ${domain}/${enc}" | tee -a "$LOG_DIR/failures.log"
    done
}

run_attack() {
    local domain=$1 enc=$2 seed=$3 gpu=$4
    local query script data_dir
    if [[ $domain == medical ]]; then
        query=pna_base; script=prototype/reconstruct_medical.py; data_dir=./assets/medical_heldout_pna/
    else
        query=sol_base; script=prototype/reconstruct_uav.py; data_dir=./assets/uav_heldout_solar/
    fi
    local model_args out
    if [[ $enc == resnet18 ]]; then
        model_args=(--model-path "malicious_models_lang_sweep/${domain}/${query}.pt" --batch-tag "ce_resnet18_s${seed}")
        out="./results/${domain}_${query}_ce_resnet18_s${seed}/metrics.json"
    else
        model_args=(--model-path "malicious_models_client_encoder/${domain}_${enc}.pt" --client-encoder "$enc"
                    --batch-tag "ce_s${seed}")
        out="./results/${domain}_${query}_enc${enc}_ce_s${seed}/metrics.json"
    fi
    if [[ -f $out ]]; then
        echo "[$(date +%H:%M:%S)] skip ${domain}/${enc}/s${seed} (done)"
        return
    fi
    echo "[$(date +%H:%M:%S)] START ${domain}/${enc}/s${seed} on GPU ${gpu}"
    "$PY" "$script" --geminio-query "$query" "${model_args[@]}" --data-dir "$data_dir" \
        --attack hfgradinv --gpu "$gpu" --max-iterations "$ITERS" --seed "$seed" \
        > "$LOG_DIR/attack_${domain}_${enc}_s${seed}.log" 2>&1 \
        || echo "[$(date +%H:%M:%S)] FAILED attack ${domain}/${enc}/s${seed}" | tee -a "$LOG_DIR/failures.log"
    echo "[$(date +%H:%M:%S)] DONE  ${domain}/${enc}/s${seed}"
}

# Worker w takes every n-th job. Jobs are ordered seed-major so the seed-42 runs for
# every encoder finish first and can be inspected before the later seeds.
worker() {
    local w=$1 gpu=$2 n=$3 i=0
    for seed in "${SEEDS[@]}"; do
        for domain in medical uav; do
            for enc in "${ENCODERS[@]}"; do
                if (( i % n == w )); then
                    run_attack "$domain" "$enc" "$seed" "$gpu"
                fi
                i=$((i + 1))
            done
        done
    done
}

if [[ $STAGE == heads || $STAGE == all ]]; then
    train_heads medical "${GPU_LIST[0]}" &
    train_heads uav "${GPU_LIST[1]:-${GPU_LIST[0]}}" &
    wait
fi

if [[ $STAGE == attack || $STAGE == all ]]; then
    n=${#GPU_LIST[@]}
    for w in "${!GPU_LIST[@]}"; do
        worker "$w" "${GPU_LIST[$w]}" "$n" &
    done
    wait
    echo "[$(date +%H:%M:%S)] client-encoder sweep complete"
fi
