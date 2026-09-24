#!/usr/bin/env bash
#SBATCH --job-name=spktag_oo_med
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --array=0-12%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_spktag_other_other_resamplemedian_pc10_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_spktag_other_other_resamplemedian_pc10_%A_%a.err

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
  scripts/cluster_glm_other_other.py \
  --patient "${PATIENTS[$SLURM_ARRAY_TASK_ID]}" \
  --region hippocampus \
  --n_jobs 8 \
  --window-tag tshift-150_tlen500_oshift+200_olen500 \
  --model-tag bert-base-bidirmax-spktag \
  --context-tag "" \
  --layer 12 \
  --n-components 10 \
  --legacy-minimal-resample-to-median \
  --results-root \
  /scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise_bertbase_bidirmax_spktag_other_other_resamplemedian_pc10
