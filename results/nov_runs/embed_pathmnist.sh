#!/usr/bin/env bash
# PathMNIST test-set embeddings for every guide VLM (data/pathmnist-{vlm}-test.pt).
cd /raid/scratch/dzimmerman2021/geminio/Geminio
export HF_HUB_OFFLINE=1
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
for vlm in biomedclip clip simclip2 simclip4 fare2 fare4 tecoa2 tecoa4; do
    [[ -f data/pathmnist-${vlm}-test.pt ]] && { echo "have $vlm"; continue; }
    echo "[$(date +%H:%M:%S)] $vlm"
    CUDA_VISIBLE_DEVICES=0 "$PY" prototype/compute_embeddings.py --domain pathmnist --vlm "$vlm" --gpu 0 --batch-size 128 \
        || echo "FAILED $vlm"
done
echo "[$(date +%H:%M:%S)] embeddings done"
