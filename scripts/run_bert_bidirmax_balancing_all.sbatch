#!/usr/bin/env bash
#SBATCH --job-name=bimax_bal
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-59%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_bidirmax_bal_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_bidirmax_bal_%A_%a.err

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
configuration_idx=$((SLURM_ARRAY_TASK_ID / 15))
patient="${PATIENTS[$patient_idx]}"

if (( configuration_idx < 2 )); then
  model=bert-base-bidirmax
  label=bertbase_bidirmax
else
  model=bert-base-bidirmax-spktag
  label=bertbase_bidirmax_spktag
fi

COMMON=(
  --patient "$patient"
  --region hippocampus
  --n_jobs 8
  --window-tag "$TAG"
  --model-tag "$model"
  --context-tag ""
  --layer 12
)

if (( configuration_idx % 2 == 0 )); then
  "$PY" -u scripts/cluster_glm_legacy_replication.py \
    "${COMMON[@]}" \
    --results-root "$BASE/clusterwise_${label}_L12_fixed_custom_softbalanced"
else
  "$PY" -u scripts/cluster_glm_legacy_replication.py \
    "${COMMON[@]}" \
    --min-matched-trials 30 \
    --min-retained-categories 2 \
    --hard-balance \
    --results-root "$BASE/clusterwise_${label}_L12_fixed_custom_hardbalanced_min30"
fi
