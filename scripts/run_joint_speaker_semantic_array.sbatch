#!/usr/bin/env bash
#SBATCH --job-name=spk_sem_joint
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/joint_speaker_semantic/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/joint_speaker_semantic/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"

patients=(
    PTYEU_task147 PTYEV_task37 PTYEY_task86 PTYEZ_task60 PTYFA_task25
    PTYFC_task28 PTYFF_task17 PTYFG_task18 PTYFI_task81 PTYFK_task40
    PTYFM_task104 PTYFP_task88 PTYFR_task91 PTYFS_task95 PTYFU_task224
)
patient="${patients[$SLURM_ARRAY_TASK_ID]}"

cd "${PROJECT}"
echo "patient=${patient} host=$(hostname) started=$(date --iso-8601=seconds)"
"${PYTHON}" -u scripts/joint_speaker_semantic_glm.py \
    --patient "${patient}" \
    --model llama-3.1-8b \
    --context_tag _ctx200 \
    --layer 14 \
    --n_components 50 \
    --n_perm 100
echo "completed=$(date --iso-8601=seconds)"
