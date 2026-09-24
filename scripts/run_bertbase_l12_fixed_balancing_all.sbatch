#!/usr/bin/env bash
#SBATCH --job-name=bert_bal_all
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-29%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_bal_all_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_bal_all_%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
TAG=tref-offset_shift-200_tlen500_oref-onset_shift+100_olen500
BASE=/scratch/aniluchavez/ConvoDATAS/SemanticGLM
PATIENTS=(
  PTYEU_task147 PTYFF_task17 PTYFG_task18 PTYFI_task81
  PTYFA_task25 PTYFK_task40 PTYEY_task86 PTYEV_task37
  PTYEZ_task60 PTYFC_task28 PTYFM_task104 PTYFP_task88
  PTYFR_task91 PTYFS_task95 PTYFU_task224
)

patient_idx=$((SLURM_ARRAY_TASK_ID % 15))
analysis_idx=$((SLURM_ARRAY_TASK_ID / 15))
patient="${PATIENTS[$patient_idx]}"

COMMON=(
  --patient "$patient"
  --region hippocampus
  --n_jobs 8
  --window-tag "$TAG"
  --model-tag bert-base
  --layer 12
)

if [[ "$analysis_idx" == "0" ]]; then
  "$PY" -u scripts/cluster_glm_legacy_replication.py \
    "${COMMON[@]}" \
    --results-root "$BASE/clusterwise_bertbase_L12_fixed_custom_softbalanced"
else
  "$PY" -u scripts/cluster_glm_legacy_replication.py \
    "${COMMON[@]}" \
    --min-matched-trials 30 \
    --min-retained-categories 2 \
    --hard-balance \
    --results-root "$BASE/clusterwise_bertbase_L12_fixed_custom_hardbalanced_min30"
fi
