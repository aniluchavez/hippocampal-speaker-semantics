#!/usr/bin/env bash
#SBATCH --job-name=joint_othoth_bert_pc30
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-12%3
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/joint_other_other_bert_pc30_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/joint_other_other_bert_pc30_%A_%a.err

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
  scripts/joint_category_interaction_other_other_glm.py \
  --patient "${PATIENTS[$SLURM_ARRAY_TASK_ID]}" \
  --region hippocampus \
  --model-tag bert-base-bidirmax \
  --context-tag "" \
  --window-tag tshift-150_tlen500_oshift+200_olen500 \
  --layer 12 \
  --n-components 30 \
  --deviation-penalty-multiplier 10 \
  --n-jobs 8 \
  --results-root \
  /scratch/aniluchavez/ConvoDATAS/SemanticGLM/joint_category_interaction_other_other_bert_pc30
