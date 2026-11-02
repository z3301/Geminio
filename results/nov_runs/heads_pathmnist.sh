#!/usr/bin/env bash
# PathMNIST heads: ResNet-18 guide heads (GPU 0) and ViT-L rigged + plain heads (GPU 2), in parallel.
cd /raid/scratch/dzimmerman2021/geminio/Geminio
export HF_HUB_OFFLINE=1
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
Q='Any colorectal histopathology image showing colorectal adenocarcinoma epithelium'
(
  for vlm in biomedclip clip simclip2 simclip4 fare2 fare4 tecoa2 tecoa4; do
    f=malicious_models_pathmnist/tum_base.pt; [[ $vlm != biomedclip ]] && f=malicious_models_pathmnist/tum_base_${vlm}.pt
    [[ -f $f ]] && { echo "have $f"; continue; }
    echo "[$(date +%H:%M:%S)] resnet head $vlm"
    CUDA_VISIBLE_DEVICES=0 "$PY" prototype/train_medical.py --dataset pathmnist --vlm "$vlm" --query "$Q" --name tum_base \
        --seed 0 --gpu 0 > "results/nov_runs/logs/head_pathmnist_resnet_${vlm}.log" 2>&1 || echo "FAILED resnet $vlm"
  done
  echo "[$(date +%H:%M:%S)] resnet heads done"
) &
(
  for enc in clip simclip2 simclip4 fare2 fare4 tecoa2 tecoa4 laion2b dinov2 in21k; do
    for mode in "" "--plain"; do
      stem="pathmnist_${enc}"; [[ -n $mode ]] && stem="${stem}_plain"
      [[ -f malicious_models_client_encoder/${stem}.pt ]] && { echo "have $stem"; continue; }
      echo "[$(date +%H:%M:%S)] $stem"
      CUDA_VISIBLE_DEVICES=2 "$PY" prototype/client_encoder.py --domain pathmnist --encoder "$enc" --gpu 0 $mode \
        > "results/nov_runs/logs/head_${stem}.log" 2>&1 || echo "FAILED $stem"
    done
  done
  echo "[$(date +%H:%M:%S)] vit heads done"
) &
wait
echo "[$(date +%H:%M:%S)] all pathmnist heads done"
