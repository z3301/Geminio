#!/usr/bin/env bash
# Train the control heads for one domain on one GPU (prototype/client_encoder.py):
#   plain  honest unweighted head on each frozen client encoder  -> {domain}_{enc}_plain.pt
#   swap   head rigged for a different query under the same attacker VLM -> {domain}_{enc}_{tag}.pt
#          medical: "Any chest X-ray showing pleural effusion"  (tag qeff; batch has 1 effusion image)
#          uav:     "aerial drone image showing an airport runway" (tag qair; batch has 2 airstrip images)
# Features come from results/client_encoder/cache, so no dataset access is needed. Existing heads
# are skipped. Usage: bash scripts/run_control_heads.sh medical 2
set -uo pipefail
cd /raid/scratch/dzimmerman2021/geminio/Geminio
DOMAIN=$1; GPU=$2
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU
ENCODERS=(clip simclip2 simclip4 fare2 fare4 tecoa2 tecoa4)
if [[ $DOMAIN == medical ]]; then QUERY="Any chest X-ray showing pleural effusion"; TAG=qeff
else QUERY="aerial drone image showing an airport runway"; TAG=qair; fi
mkdir -p results/plain_head/logs results/query_swap/logs
for enc in "${ENCODERS[@]}"; do
    if [[ ! -f malicious_models_client_encoder/${DOMAIN}_${enc}_plain.pt ]]; then
        echo "[$(date +%H:%M:%S)] plain ${DOMAIN}/${enc} on GPU ${GPU}"
        "$PY" prototype/client_encoder.py --domain "$DOMAIN" --encoder "$enc" --plain --gpu 0 \
            > "results/plain_head/logs/head_${DOMAIN}_${enc}_plain.log" 2>&1 \
            || echo "FAILED plain ${DOMAIN}/${enc}" | tee -a results/plain_head/logs/failures.log
    fi
    if [[ ! -f malicious_models_client_encoder/${DOMAIN}_${enc}_${TAG}.pt ]]; then
        echo "[$(date +%H:%M:%S)] swap ${DOMAIN}/${enc} (${TAG}) on GPU ${GPU}"
        "$PY" prototype/client_encoder.py --domain "$DOMAIN" --encoder "$enc" --query "$QUERY" --query-tag "$TAG" --gpu 0 \
            > "results/query_swap/logs/head_${DOMAIN}_${enc}_${TAG}.log" 2>&1 \
            || echo "FAILED swap ${DOMAIN}/${enc}" | tee -a results/query_swap/logs/failures.log
    fi
done
echo "[$(date +%H:%M:%S)] ${DOMAIN} control heads done"
