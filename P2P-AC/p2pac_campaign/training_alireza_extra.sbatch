#!/usr/bin/env bash
#SBATCH --job-name=p2pac-training-alireza_extra
#SBATCH --account=naiss2026-4-2061-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-1%2
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1645/personal/mohsena/P2P-AC_UnitreeGo2/P2P-AC/p2pac_campaign/logs/training_alireza_extra/%A_%a.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-1645/personal/mohsena/P2P-AC_UnitreeGo2/P2P-AC/p2pac_campaign/logs/training_alireza_extra/%A_%a.err

set -euo pipefail
cd '/nobackup/proj/disk/naiss2024-22-1645/personal/mohsena/P2P-AC_UnitreeGo2/P2P-AC'
export P2PAC_PIPBASE="$HOME/.p2pac_pipbase/aarch64"
source scripts/container.sh

exec bash p2pac_campaign/run_group.sh \
    p2pac_campaign/jobs/training_alireza_extra.txt \
    "$SLURM_ARRAY_TASK_ID" 5 60
