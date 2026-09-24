#!/usr/bin/env bash
#SBATCH --job-name=othoth_rcross_pc10
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-12%3
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/other_other_rcross_pc10_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/other_other_rcross_pc10_%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics

PATIENTS=(
  PTYEU_task147
  PTYEY_task86
  PTYFA_task25
  PTYFC_task28
  PTYFF_task17
  PTYFG_task18
  PTYFK_task40
  PTYFM_task104
  PTYFP_task88
  PTYFR_task91
  PTYFS_task95
  PTYFU_task224
  PTYEZ_task60
)

/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/other_other_rcross_reliability.py \
  --patient "${PATIENTS[$SLURM_ARRAY_TASK_ID]}" \
  --region hippocampus \
  --model-tag bert-base-bidirmax \
  --context-tag "" \
  --window-tag tshift-150_tlen500_oshift+200_olen500 \
  --layer 12 \
  --n-components 10 \
  --n-null 100 \
  --n-half-splits 100 \
  --n-jobs 8 \
  --results-root \
  /scratch/aniluchavez/ConvoDATAS/SemanticGLM/other_other_rcross_reliability_bert_pc10
