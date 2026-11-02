#!/usr/bin/env bash
# Score groups B-E (ViT-L client runs) whenever enough of them exist; safe to re-run.
#   results/client_encoder_nov/        rigged heads, all encoders, domains medical/uav/pathmnist (vs CLIP)
#   results/client_encoder_plain_nov/  plain vs rigged contrast, same scope
# then the ViT-L private-batch profile, the reference comparison and the backbone figure.
set -uo pipefail
cd /raid/scratch/dzimmerman2021/geminio/Geminio
export HF_HUB_OFFLINE=1
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-3}
CUDA_VISIBLE_DEVICES=$GPU "$PY" scripts/summarize_ci.py --experiment client_encoder --domains medical uav pathmnist \
    --out-dir results/client_encoder_nov --gpu 0
CUDA_VISIBLE_DEVICES=$GPU "$PY" scripts/summarize_ci.py --experiment client_encoder_plain --domains medical uav pathmnist \
    --out-dir results/client_encoder_plain_nov --gpu 0
"$PY" prototype/compare_to_baselines.py
RUNS=results/client_encoder_plain_nov/runs.csv "$PY" - <<'EOF'
import os, runpy, sys
sys.argv = ['plot_backbone.py', '--domain', 'uav']
import prototype.plot_backbone as pb
pb.RUNS = os.environ['RUNS']
pb.main()
EOF
echo "[$(date '+%m-%d %H:%M')] rest scored"
