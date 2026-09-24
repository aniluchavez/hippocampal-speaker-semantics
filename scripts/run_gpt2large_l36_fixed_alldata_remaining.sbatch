#!/usr/bin/env bash
#SBATCH --job-name=gpt2_all_all
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-13%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/gpt2_alldata_remaining_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/gpt2_alldata_remaining_%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
PATIENTS=(
  PTYEU_task147 PTYFF_task17 PTYFG_task18 PTYFI_task81
  PTYFA_task25 PTYFK_task40 PTYEY_task86 PTYEV_task37
  PTYFC_task28 PTYFM_task104 PTYFP_task88 PTYFR_task91
  PTYFS_task95 PTYFU_task224
)
patient="${PATIENTS[$SLURM_ARRAY_TASK_ID]}"
/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/cluster_glm_legacy_replication.py \
  --patient "$patient" \
  --region hippocampus \
  --n_jobs 8 \
  --window-tag tref-offset_shift-200_tlen500_oref-onset_shift+100_olen500 \
  --model-tag gpt2-large \
  --layer 36 \
  --no-balance \
  --results-root /scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise_gpt2large_L36_fixed_custom_alldata
