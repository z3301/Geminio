#!/usr/bin/env bash
cd /raid/scratch/dzimmerman2021/geminio/Geminio
while ! grep -q 'queue complete' results/nov_runs/queueA.log; do sleep 60; done
GPU=3 bash results/nov_runs/score_group_a.sh
