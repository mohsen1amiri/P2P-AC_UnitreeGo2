#!/bin/bash -l
#SBATCH --job-name=p2pac-kamiak-training
#SBATCH --partition=camas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6-23:00:00
#SBATCH --array=0-4%2
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/lab/hosseinzadeh/personal/mohsen.amiri/P2P-AC/P2P-AC/p2pac_campaign/logs/training_kamiak/%A_%a.out
#SBATCH --error=/data/lab/hosseinzadeh/personal/mohsen.amiri/P2P-AC/P2P-AC/p2pac_campaign/logs/training_kamiak/%A_%a.err

set -euo pipefail
module load StdEnv apptainer/1.3.4
cd '/data/lab/hosseinzadeh/personal/mohsen.amiri/P2P-AC/P2P-AC'

export P2PAC_PIPBASE="$HOME/.p2pac_pipbase/x86_64"
export P2PAC_CACHE_ROOT="$HOME/.cache/p2pac-isaac"

source scripts/container.sh

exec bash p2pac_campaign/run_group.sh \
    p2pac_campaign/jobs/training_kamiak.txt \
    "$SLURM_ARRAY_TASK_ID" 5 60
