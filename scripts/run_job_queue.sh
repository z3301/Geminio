#!/usr/bin/env bash
# Dynamic GPU job queue. Each worker (one per GPU) claims the next line of a jobs file
# under flock, pins it to that GPU with CUDA_VISIBLE_DEVICES (the script sees it as --gpu 0), and runs it. A job whose result dir already has
# metrics.json is skipped, so the queue can be restarted: delete <queue>/next to rescan
# from the top. Optionally waits until no process matches WAIT_FOR before starting.
#
# Jobs file format (scripts/make_heldout_jobs.py): tab-separated
#     <result dir> <python script> <arg> <arg> ...
#
# Usage:
#   GPUS="3 5 6 7" WAIT_FOR="run_client_encoder.sh attack" \
#       bash scripts/run_job_queue.sh results/heldout_rerun/jobs.tsv
set -uo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio

JOBS=$1
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
read -r -a GPU_LIST <<< "${GPUS:-3 5 6 7}"
QUEUE_DIR="$(dirname "$JOBS")/queue"
mkdir -p "$QUEUE_DIR/logs"
COUNTER="$QUEUE_DIR/next"
LOCK="$QUEUE_DIR/lock"
[[ -f $COUNTER ]] || echo 1 > "$COUNTER"
export HF_HUB_OFFLINE=1

if [[ -n ${WAIT_FOR:-} ]]; then
    # Exclude this script's own process tree: the WAIT_FOR text appears in our launch
    # command line, so an unfiltered pgrep -f would match ourselves and never return.
    echo "[$(date '+%m-%d %H:%M')] waiting for '${WAIT_FOR}' to finish"
    while pgrep -f "$WAIT_FOR" | grep -v -x -e "$$" -e "$PPID" | grep -q .; do sleep 300; done
fi

TOTAL=$(wc -l < "$JOBS")
NEXT=$(cat "$COUNTER")
if (( NEXT > TOTAL )); then
    echo "ERROR: ${COUNTER} is at ${NEXT} but ${JOBS} has ${TOTAL} jobs; every job would be skipped." >&2
    echo "       Delete ${COUNTER} to rescan from the top (jobs with metrics.json are skipped anyway)." >&2
    exit 2
fi
echo "[$(date '+%m-%d %H:%M')] starting ${TOTAL} jobs on GPUs ${GPU_LIST[*]} (counter at ${NEXT})"

claim() {
    (
        flock 9
        local n
        n=$(cat "$COUNTER")
        echo $((n + 1)) > "$COUNTER"
        echo "$n"
    ) 9> "$LOCK"
}

worker() {
    local gpu=$1 n line out script name
    while true; do
        n=$(claim)
        (( n > TOTAL )) && break
        line=$(sed -n "${n}p" "$JOBS")
        IFS=$'\t' read -r -a fields <<< "$line"
        out=${fields[0]}
        script=${fields[1]}
        name=$(basename "$out")
        if [[ -f "$out/metrics.json" ]]; then
            echo "[$(date '+%m-%d %H:%M')] skip  ${n}/${TOTAL} ${name} (done)"
            continue
        fi
        echo "[$(date '+%m-%d %H:%M')] START ${n}/${TOTAL} ${name} on GPU ${gpu}"
        CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$script" "${fields[@]:2}" --gpu 0 > "$QUEUE_DIR/logs/${name}.log" 2>&1 \
            || echo "[$(date '+%m-%d %H:%M')] FAILED ${n} ${name}" | tee -a "$QUEUE_DIR/failures.log"
        echo "[$(date '+%m-%d %H:%M')] DONE  ${n}/${TOTAL} ${name}"
    done
}

for gpu in "${GPU_LIST[@]}"; do
    worker "$gpu" &
done
wait
echo "[$(date '+%m-%d %H:%M')] queue complete"
