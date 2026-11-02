#!/usr/bin/env bash
# When every PathMNIST ViT-L head exists: regenerate the job lists, assemble groups B-E into their own
# queue directory (separate counter from group A), and start that queue once group A's queue has exited.
cd /raid/scratch/dzimmerman2021/geminio/Geminio
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
while ! grep -q 'all pathmnist heads done' results/nov_runs/logs/heads_pathmnist.log; do sleep 30; done
"$PY" scripts/make_nov_jobs.py
mkdir -p results/nov_runs/rest
cat results/nov_runs/uav_comparators.tsv results/nov_runs/pathmnist_vit.tsv \
    results/nov_runs/medical_comparators.tsv results/nov_runs/plain_tail.tsv > results/nov_runs/rest/jobs.tsv
echo "[$(date '+%m-%d %H:%M')] rest queue: $(wc -l < results/nov_runs/rest/jobs.tsv) jobs; waiting for group A"
GPUS="0 1 2 3" WAIT_FOR="pathmnist_resnet.tsv" bash scripts/run_job_queue.sh results/nov_runs/rest/jobs.tsv
