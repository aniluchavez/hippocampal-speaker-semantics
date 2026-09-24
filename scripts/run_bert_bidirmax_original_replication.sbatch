#!/usr/bin/env bash
#SBATCH --job-name=bert_origrep
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-9%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_origrep_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_origrep_%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics

PATIENTS=(
  PTYEU_task147 PTYFF_task17 PTYFG_task18 PTYFI_task81
  PTYFA_task25 PTYEY_task86 PTYEV_task37 PTYEZ_task60
  PTYFK_task40 PTYFC_task28
)

/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/cluster_glm_legacy_replication.py \
  --patient "${PATIENTS[$SLURM_ARRAY_TASK_ID]}" \
  --region hippocampus \
  --n_jobs 8 \
  --window-tag tshift-150_tlen500_oshift+200_olen500 \
  --model-tag bert-base-bidirmax \
  --context-tag "" \
  --layer 12 \
  --results-root \
  /scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise_bertbase_bidirmax_original10_softbalanced_replication
