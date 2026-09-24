#!/usr/bin/env bash
#SBATCH --job-name=bert_orig5
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_orig5_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_orig5_%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics

PATIENTS=(
  PTYFM_task104 PTYFP_task88 PTYFR_task91 PTYFS_task95 PTYFU_task224
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
