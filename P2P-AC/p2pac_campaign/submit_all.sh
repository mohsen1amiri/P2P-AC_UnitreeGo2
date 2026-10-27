#!/usr/bin/env bash
# Submit the full P2P-AC campaign in stage order with Slurm dependencies.
# Runs 153 scientific jobs (+290 independent evaluation jobs) across three accounts.
# A completed run is skipped by p2pac.train; interrupted runs resume from checkpoints.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p p2pac_campaign/submission_logs
# Ensure Slurm log paths exist before submitting.
for sb in p2pac_campaign/*.sbatch; do
  mkdir -p "p2pac_campaign/logs/$(basename "$sb" .sbatch)"
done
JOURNAL="p2pac_campaign/submission_logs/$(date +%Y%m%d_%H%M%S)_submitted.txt"
submit() {
  local id
  id=$(sbatch --parsable "$@")
  id="${id%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "Unexpected Slurm response: $id" >&2; exit 1; }
  printf '%s %s\n' "$id" "${*: -1}" | tee -a "$JOURNAL" >&2
  printf '%s\n' "$id"
}
# Submit independent REAP reference evaluations immediately.
for acc in mohsen sindri alireza; do
  submit "p2pac_campaign/reference_${acc}.sbatch" >/dev/null
done
# Training: 5 independent learners per GPU; 2 GPUs maximum per account at a time.
train_ids=()
for acc in mohsen sindri alireza; do
  train_ids+=("$(submit "p2pac_campaign/training_${acc}.sbatch")")
done
# Evaluate the 101 saved snapshots on 10 independent episodes each.
# Then evaluate the final 5M-step policy on 100 independent episodes per seed.
names=(mohsen sindri alireza)
for i in 0 1 2; do
  snapshot_id=$(submit --dependency="afterok:${train_ids[$i]}" "p2pac_campaign/snapshots_${names[$i]}.sbatch")
  submit --dependency="afterok:${snapshot_id}" "p2pac_campaign/final_eval_${names[$i]}.sbatch" >/dev/null
done
# BC uses the protected planner replay buffers collected by the main P2P-SAC runs.
submit --dependency="afterok:${train_ids[0]}:${train_ids[1]}:${train_ids[2]}" \
  "p2pac_campaign/bc_alireza.sbatch" >/dev/null
printf '\nSubmitted all campaign stages; IDs recorded in %s\n' "$JOURNAL"
