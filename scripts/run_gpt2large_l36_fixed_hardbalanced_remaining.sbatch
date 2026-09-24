#!/usr/bin/env bash
#SBATCH --job-name=gpt2_bal_all
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --array=0-13%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/gpt2_bal_remaining_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/gpt2_bal_remaining_%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
TAG=tref-offset_shift-200_tlen500_oref-onset_shift+100_olen500
PATIENTS=(
  PTYEU_task147 PTYFF_task17 PTYFG_task18 PTYFI_task81
  PTYFA_task25 PTYFK_task40 PTYEY_task86 PTYEV_task37
  PTYFC_task28 PTYFM_task104 PTYFP_task88 PTYFR_task91
  PTYFS_task95 PTYFU_task224
)
patient="${PATIENTS[$SLURM_ARRAY_TASK_ID]}"

"$PY" -u scripts/generate_spike_windows.py \
  --mode fixed \
  --patients "$patient" \
  --target_ref_point offset \
  --target_shift -200 \
  --target_window_length 500 \
  --other_shift 100 \
  --other_window_length 500 \
  --out_tag "$TAG"

"$PY" -u scripts/cluster_glm_legacy_replication.py \
  --patient "$patient" \
  --region hippocampus \
  --n_jobs 8 \
  --window-tag "$TAG" \
  --model-tag gpt2-large \
  --layer 36 \
  --min-matched-trials 30 \
  --min-retained-categories 2 \
  --hard-balance \
  --results-root /scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise_gpt2large_L36_fixed_custom_hardbalanced_min30
