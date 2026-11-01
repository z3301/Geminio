#!/usr/bin/env bash
# UAV half of the temperature sweep (control C), end to end: train the 9 UAV heads (rep 1 on the
# first GPU, rep 2 on the second), regenerate the job lists, run the 27 UAV attack jobs on all the
# listed GPUs (medical jobs are skipped because their metrics.json exist), then score the sweep.
# Launch detached: GPUS="0 1 2 3" setsid nohup bash scripts/run_uav_tempsweep.sh > results/temp_sweep/uav_pipeline.log 2>&1 < /dev/null &
set -uo pipefail
cd /raid/scratch/dzimmerman2021/geminio/Geminio
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
export HF_HUB_OFFLINE=1
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"
echo "[$(date '+%m-%d %H:%M')] UAV temperature heads on GPUs ${GPU_LIST[0]} (rep 1) and ${GPU_LIST[1]:-${GPU_LIST[0]}} (rep 2)"
CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} $PY prototype/run_temperature_train.py --domain uav --rep 1 --gpu 0 > results/temp_sweep/heads_uav_rep1.log 2>&1 &
CUDA_VISIBLE_DEVICES=${GPU_LIST[1]:-${GPU_LIST[0]}} $PY prototype/run_temperature_train.py --domain uav --rep 2 --gpu 0 > results/temp_sweep/heads_uav_rep2.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] heads: $(ls malicious_models_tempsweep/uav/*.pt 2>/dev/null | wc -l) / 9"
$PY scripts/make_rerun_jobs.py | grep -v "^    "
rm -f results/temp_sweep/queue/next
GPUS="${GPU_LIST[*]}" bash scripts/run_job_queue.sh results/temp_sweep/jobs.tsv
echo "[$(date '+%m-%d %H:%M')] attack queue finished; scoring"
CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} $PY prototype/analyze_temperature_sweep.py --gpu 0 > results/temp_sweep/score_uav.log 2>&1
echo "[$(date '+%m-%d %H:%M')] UAV temperature sweep pipeline complete (rc=$?)"
