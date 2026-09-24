#!/usr/bin/env bash
#SBATCH --job-name=glm_wdur
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/cluster_glm_worddur_offset_array/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/cluster_glm_worddur_offset_array/%A_%a.err

set -euo pipefail

PROJECT_ROOT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"

PATIENTS=(
    PTYEU_task147
    PTYFF_task17
    PTYFG_task18
    PTYFI_task81
    PTYFA_task25
    PTYFK_task40
    PTYEY_task86
    PTYEV_task37
    PTYEZ_task60
    PTYFC_task28
    PTYFM_task104
    PTYFP_task88
    PTYFR_task91
    PTYFS_task95
    PTYFU_task224
)

patient="${PATIENTS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "$PROJECT_ROOT/logs/cluster_glm_worddur_offset_array"
cd "$PROJECT_ROOT"

echo "Starting $patient on $(hostname) at $(date --iso-8601=seconds)"
"$PYTHON" -u scripts/cluster_glm_worddur_offset.py --patient "$patient"
echo "Completed $patient at $(date --iso-8601=seconds)"
