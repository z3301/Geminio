#!/usr/bin/env bash
# Control B for UAV: compute any missing per-VLM UAV embeddings (prototype/compute_embeddings.py writes
# uav_{vlm}_embeddings_<scenes>.pt under the UAVScenes root), then re-run the retrieval analysis for
# both domains. Launch detached with CUDA_VISIBLE_DEVICES set.
set -uo pipefail
cd /raid/scratch/dzimmerman2021/geminio/Geminio
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
export HF_HUB_OFFLINE=1
ROOT=/raid/scratch/dzimmerman2021/uavscenes
for vlm in clip simclip2 simclip4 fare2 fare4 tecoa2 tecoa4; do
    f="$ROOT/uav_${vlm}_embeddings_interval5_AMtown01_interval5_HKairport01.pt"
    if [[ -f $f ]]; then echo "[$(date '+%m-%d %H:%M')] have $vlm"; continue; fi
    echo "[$(date '+%m-%d %H:%M')] computing UAV embeddings for $vlm"
    $PY prototype/compute_embeddings.py --domain uav --vlm "$vlm" --gpu 0 > "results/complementarity/logs/uav_embed_${vlm}.log" 2>&1 \
        || echo "FAILED embeddings $vlm"
done
$PY prototype/analyze_vlm_retrieval.py --gpu 0 > results/complementarity/logs/vlm_retrieval_uav.log 2>&1
echo "[$(date '+%m-%d %H:%M')] retrieval analysis rc=$?"
