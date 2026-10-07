#!/bin/bash
# Run every command of a jobs.txt file one after another on the current machine (no SLURM).
#
#   bash scripts/run_local.sh sweeps/smoke_kinematic/jobs.txt              # kinematic runs, local Python
#   bash scripts/run_local.sh sweeps/main/jobs.txt --container             # Isaac runs, inside the container
#
# Without --container, 'python' is the interpreter in $P2PAC_PY (default: python). With --container, each command
# runs through p2pac_run (scripts/container.sh). A failed run does not stop the others; the exit code is 1 if any
# run failed. Finished runs are skipped and unfinished ones resume, so the script can simply be started again.
set -uo pipefail
JOBS_FILE="$1"
MODE="${2:-}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$MODE" == "--container" ]]; then
  source "$REPO/scripts/container.sh"
  RUNNER="p2pac_run python"
else
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
  RUNNER="${P2PAC_PY:-python}"
fi
FAILED=0
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  echo "[run_local] $(date '+%F %T') $line"
  eval "$RUNNER ${line#python }" || { echo "[run_local] FAILED: $line"; FAILED=1; }
done < "$JOBS_FILE"
exit $FAILED
