#!/usr/bin/env bash
#SBATCH --job-name=gpt2_soft_stab
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/gpt2_soft_stab_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/gpt2_soft_stab_%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
PATIENTS=(
  PTYEU_task147 PTYFF_task17 PTYFG_task18 PTYFI_task81
  PTYFA_task25 PTYFK_task40 PTYEY_task86 PTYEV_task37
  PTYEZ_task60 PTYFC_task28 PTYFM_task104 PTYFP_task88
  PTYFR_task91 PTYFS_task95 PTYFU_task224
)
patient="${PATIENTS[$SLURM_ARRAY_TASK_ID]}"
/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/gpt2_fixed_beta_fold_stability.py \
  --patient "$patient" \
  --results-root /scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise_gpt2large_L36_fixed_custom_softbalanced \
  --balance-mode soft \
  --min-trials 10
