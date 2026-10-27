#!/usr/bin/env bash
# P2P-AC campaign job-group runner. One GPU shared by at most K independent simulations.
# Safe on 3-day Arrhenius jobs: USR1 requests a checkpoint from trainers, then requeues.
set -uo pipefail
JOBS_FILE="$1"; GROUP="$2"; K="$3"; STAGGER="$4"
mapfile -t ALL < "$JOBS_FILE"
START=$((GROUP*K))
declare -a PIDS=()
REQUEST_REQUEUE=0

children_recursive() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null); do
    printf '%s\n' "$child"
    children_recursive "$child"
  done
}
signal_jobs() {
  local sig="$1" top child cmd proc_comm
  REQUEST_REQUEUE=1
  echo "[campaign] caught $sig at $(date -Is), requesting graceful termination" >&2
  for top in "${PIDS[@]}"; do
    for child in $(children_recursive "$top"); do
      cmd=$(ps -o args= -p "$child" 2>/dev/null || true)
      proc_comm=$(ps -o comm= -p "$child" 2>/dev/null || true)
      [[ "$proc_comm" == python* ]] || continue
      if [[ "$cmd" == *'p2pac.train'* ]]; then
        kill -USR1 "$child" 2>/dev/null || true
      elif [[ "$cmd" == *'p2pac.evaluate'* ]]; then
        kill -TERM "$child" 2>/dev/null || true
      fi
    done
  done
}
trap 'signal_jobs USR1' USR1
trap 'signal_jobs TERM' TERM

for ((i=START; i<START+K && i<${#ALL[@]}; i++)); do
  if ((REQUEST_REQUEUE)); then break; fi
  [[ -z "${ALL[$i]}" ]] && continue
  if ((i>START && STAGGER>0)); then
    sleep "$STAGGER" & wait $! || true
  fi
  echo "[campaign] start task $i at $(date -Is)"
  (
    eval "p2pac_run ${ALL[$i]}"
  ) &
  PIDS+=("$!")
done

FAILED=0; PREEMPTED=0
for pid in "${PIDS[@]}"; do
  rc=0
  # `wait` can be interrupted by the USR1 trap while the child keeps running.
  while kill -0 "$pid" 2>/dev/null; do
    wait "$pid" || rc=$?
    if ((rc > 128 && REQUEST_REQUEUE)); then rc=0; fi
  done
  wait "$pid" || rc=$?
  echo "[campaign] pid $pid exit=$rc at $(date -Is)"
  if ((rc==3)); then PREEMPTED=1
  elif ((rc!=0)); then FAILED=1; fi
done
if ((REQUEST_REQUEUE || PREEMPTED)); then
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    TASK_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    echo "[campaign] requeue $TASK_ID"
    scontrol requeue "$TASK_ID" || exit 1
    exit 0
  fi
fi
exit "$FAILED"
