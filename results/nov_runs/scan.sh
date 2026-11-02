#!/usr/bin/env bash
cd /raid/scratch/dzimmerman2021/geminio/Geminio
export HF_HUB_OFFLINE=1
while ! grep -q 'embeddings done' results/nov_runs/logs/embed_pathmnist.log; do sleep 15; done
CUDA_VISIBLE_DEVICES=0 /raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python prototype/scan_medmnist_retrieval.py --domain pathmnist --gpu 0
echo SCAN_DONE
