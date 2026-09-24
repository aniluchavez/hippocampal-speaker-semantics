#!/usr/bin/env bash
#SBATCH --job-name=trans_top2_pc30
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/transferred_other_other_top2_pc30_excl_yfg_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/transferred_other_other_top2_pc30_excl_yfg_%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics

PATIENTS=(
  PTYEU_task147
  PTYEY_task86
  PTYFA_task25
  PTYFC_task28
  PTYFF_task17
  PTYFK_task40
)

/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/cluster_glm_transferred_other_other_top2.py \
  --patient "${PATIENTS[$SLURM_ARRAY_TASK_ID]}" \
  --region hippocampus \
  --n-jobs 8 \
  --n-components 30 \
  --results-root \
  /scratch/aniluchavez/ConvoDATAS/SemanticGLM/transferred_bert_other_other_top2_pc30_excl_yfg
