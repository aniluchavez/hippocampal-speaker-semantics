#!/usr/bin/env bash
#SBATCH --job-name=bert_beta100
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_beta_stability_100_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_beta_stability_100_%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics

PATIENTS=(
  PTYEU_task147 PTYFF_task17 PTYFG_task18 PTYFI_task81
  PTYFA_task25 PTYEY_task86 PTYEV_task37 PTYEZ_task60
  PTYFK_task40 PTYFC_task28 PTYFM_task104 PTYFP_task88
  PTYFR_task91 PTYFS_task95 PTYFU_task224
)

/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/beta_variance_learning_curve.py \
  --patient "${PATIENTS[$SLURM_ARRAY_TASK_ID]}" \
  --region hippocampus \
  --repeats 25 \
  --n-jobs 8 \
  --model-tag bert-base-bidirmax \
  --context-tag "" \
  --layer 12 \
  --n-components 10 \
  --window-tag tshift-150_tlen500_oshift+200_olen500 \
  --sizes 100 \
  --results-root /scratch/aniluchavez/ConvoDATAS/SemanticGLM/beta_stability_bert_bidirmax_pc10_size100
