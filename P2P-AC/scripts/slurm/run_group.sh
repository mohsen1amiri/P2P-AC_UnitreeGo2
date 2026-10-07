#!/bin/bash
# Run one group of commands (one SLURM array task) on the same GPU.
#
#   run_group.sh <jobs.txt> <group index> <runs per group> <stagger seconds>
#
# * lines [index*k, index*k + k) of jobs.txt are started in the background, staggered by <stagger> seconds
#   (Isaac Sim instances started at the same instant compete for the shader cache);
# * SIGUSR1/SIGTERM (sent by SLURM before the time limit, see --signal in submit.sbatch) is forwarded to the
#   python processes of every run (found in the process tree: Isaac Sim's python.sh is a bash wrapper, so the
#   background PID itself must not be signalled). Each run writes a checkpoint and exits with code 3;
# * if any run was preempted, the array task is requeued; the identical commands then resume automatically
#   (finished runs are skipped, unfinished ones continue from their latest checkpoint).
set -uo pipefail
JOBS_FILE="$1"; GROUP="$2"; K="$3"; STAGGER="$4"

ALL=()
while IFS= read -r line || [[ -n "$line" ]]; do ALL+=("$line"); done < "$JOBS_FILE"
START=$(( GROUP * K ))
PIDS=()

descendants() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do echo "$c"; descendants "$c"; done; }
forward() {
  echo "[run_group] $(date '+%Y-%m-%dT%H:%M:%S') forwarding SIG$1"
  local pid d
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do
    for d in $(descendants "$pid"); do
      if ps -o args= -p "$d" 2>/dev/null | grep -q "p2pac.train"; then kill -s "$1" "$d" 2>/dev/null || true; fi
    done
  done
}
trap 'forward USR1' USR1
trap 'forward TERM' TERM

for (( i=START; i<START+K && i<${#ALL[@]}; i++ )); do
  CMD="${ALL[$i]}"
  [[ -z "$CMD" ]] && continue
  if (( i > START )); then sleep "$STAGGER" & wait $!; fi
  echo "[run_group] $(date '+%Y-%m-%dT%H:%M:%S') line $i: $CMD"
  eval "p2pac_run ${CMD}" &
  PIDS+=($!)
done

PREEMPTED=0; FAILED=0
for pid in ${PIDS[@]+"${PIDS[@]}"}; do
  while kill -0 "$pid" 2>/dev/null; do wait "$pid"; done   # 'wait' returns early when a trapped signal arrives
  wait "$pid"; code=$?
  echo "[run_group] $(date '+%Y-%m-%dT%H:%M:%S') pid $pid exited with $code"
  if (( code == 3 )); then PREEMPTED=1; elif (( code != 0 )); then FAILED=1; fi
done

if (( PREEMPTED )) && [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  echo "[run_group] requeueing ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
exit $FAILED
