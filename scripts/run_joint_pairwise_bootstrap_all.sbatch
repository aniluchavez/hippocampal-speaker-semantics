#!/usr/bin/env bash
#SBATCH --job-name=joint_pair
#SBATCH --partition=guppy
#SBATCH --array=0-14
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/joint_pair_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/joint_pair_%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics

PATIENTS=(
  PTYEU_task147 PTYFF_task17 PTYFG_task18 PTYFI_task81
  PTYFA_task25 PTYFK_task40 PTYEY_task86 PTYEV_task37
  PTYEZ_task60 PTYFC_task28 PTYFM_task104 PTYFP_task88
  PTYFR_task91 PTYFS_task95 PTYFU_task224
)

/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/joint_reduced_model_bootstrap.py \
  --patient "${PATIENTS[$SLURM_ARRAY_TASK_ID]}" \
  --n-bootstraps 1000 \
  --n-jobs 8 \
  --n-components 10 \
  --deviation-penalty-multiplier 10 \
  --results-root \
  /scratch/aniluchavez/ConvoDATAS/SemanticGLM/joint_category_interaction_gpt2large_L36
