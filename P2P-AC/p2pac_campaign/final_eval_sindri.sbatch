#!/bin/bash
#SBATCH --job-name=p2pac-final_eval-sindri
#SBATCH --account=naiss2026-3-292-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1645/personal/mohsena/P2P-AC_UnitreeGo2/P2P-AC/p2pac_campaign/logs/final_eval_sindri/%A_%a.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-1645/personal/mohsena/P2P-AC_UnitreeGo2/P2P-AC/p2pac_campaign/logs/final_eval_sindri/%A_%a.err
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
# Pre-generated for verified P2P-AC repository; do not change scientific arguments.
set -euo pipefail
cd '/nobackup/proj/disk/naiss2024-22-1645/personal/mohsena/P2P-AC_UnitreeGo2/P2P-AC'
export P2PAC_PIPBASE=/home/mohsena/.p2pac_pipbase/aarch64
source scripts/container.sh
exec bash p2pac_campaign/run_group.sh   '/nobackup/proj/disk/naiss2024-22-1645/personal/mohsena/P2P-AC_UnitreeGo2/P2P-AC/p2pac_campaign/jobs/final_eval_sindri.txt'   "$SLURM_ARRAY_TASK_ID" 5 45
