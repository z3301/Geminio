#!/usr/bin/env bash
# Score group A (PathMNIST, ResNet-18) once results/nov_runs/queueA.log says "queue complete":
# held-out style summary, uninformative-set reference for the PathMNIST batch, reference comparison, and the
# retrieval-vs-attack-strength correlation across the 8 guides.
set -uo pipefail
cd /raid/scratch/dzimmerman2021/geminio/Geminio
export HF_HUB_OFFLINE=1
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-3}
mkdir -p results/nov_runs/pathmnist_resnet
CUDA_VISIBLE_DEVICES=$GPU "$PY" scripts/summarize_heldout.py --jobs results/nov_runs/pathmnist_resnet.tsv \
    --out-dir results/nov_runs/pathmnist_resnet --gpu 0
CUDA_VISIBLE_DEVICES=$GPU "$PY" prototype/floor_baselines.py --gpu 0
"$PY" prototype/compare_to_baselines.py
echo "[$(date '+%m-%d %H:%M')] group A scored"
