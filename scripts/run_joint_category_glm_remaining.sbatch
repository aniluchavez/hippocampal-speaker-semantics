#!/usr/bin/env bash
#SBATCH --job-name=joint_cat_all
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-13%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/joint_remaining_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/joint_remaining_%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
PATIENTS=(
  PTYEU_task147 PTYFF_task17 PTYFG_task18 PTYFI_task81
  PTYFA_task25 PTYFK_task40 PTYEY_task86 PTYEV_task37
  PTYFC_task28 PTYFM_task104 PTYFP_task88 PTYFR_task91
  PTYFS_task95 PTYFU_task224
)
patient="${PATIENTS[$SLURM_ARRAY_TASK_ID]}"
/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/joint_category_interaction_glm.py \
  --patient "$patient" \
  --n-components 10 \
  --deviation-penalty-multiplier 10 \
  --n-jobs 8 \
  --results-root /scratch/aniluchavez/ConvoDATAS/SemanticGLM/joint_category_interaction_gpt2large_L36
