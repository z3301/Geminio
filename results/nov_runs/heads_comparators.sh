#!/usr/bin/env bash
# Rigged + plain Geminio heads for the non-robust comparator encoders on UAV and ChestMNIST.
cd /raid/scratch/dzimmerman2021/geminio/Geminio
export HF_HUB_OFFLINE=1
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
for enc in laion2b dinov2 in21k; do
  for domain in uav medical; do
    for mode in "" "--plain"; do
      stem="${domain}_${enc}"; [[ -n $mode ]] && stem="${stem}_plain"
      [[ -f malicious_models_client_encoder/${stem}.pt ]] && { echo "have $stem"; continue; }
      echo "[$(date +%H:%M:%S)] $stem"
      CUDA_VISIBLE_DEVICES=1 "$PY" prototype/client_encoder.py --domain "$domain" --encoder "$enc" --gpu 0 $mode \
        > "results/nov_runs/logs/head_${stem}.log" 2>&1 || echo "FAILED $stem"
    done
  done
done
echo "[$(date +%H:%M:%S)] comparator heads done"
