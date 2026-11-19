#!/usr/bin/env bash
#SBATCH --job-name=p2pac-eval-ready
#SBATCH --account=naiss2026-3-292-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-15%2
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1645/personal/mohsena/P2P-AC_UnitreeGo2/P2P-AC/p2pac_campaign/logs/final_eval_ready_20261119/%A_%a.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-1645/personal/mohsena/P2P-AC_UnitreeGo2/P2P-AC/p2pac_campaign/logs/final_eval_ready_20261119/%A_%a.err

set -euo pipefail
cd '/nobackup/proj/disk/naiss2024-22-1645/personal/mohsena/P2P-AC_UnitreeGo2/P2P-AC'
export P2PAC_PIPBASE="$HOME/.p2pac_pipbase/aarch64"
source scripts/container.sh

exec bash p2pac_campaign/run_group.sh \
    p2pac_campaign/jobs/final_eval_ready_20261119.txt \
    "$SLURM_ARRAY_TASK_ID" 5 45
