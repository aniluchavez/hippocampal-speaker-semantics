#!/usr/bin/env bash
#SBATCH --job-name=bertspk_layers
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-194%8
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_causal_spktag_m150_p200_layers_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_causal_spktag_m150_p200_layers_%A_%a.err

set -euo pipefail

cd /scratch/aniluchavez/hippocampal-speaker-semantics
mkdir -p logs

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
TAG=tshift-150_tlen500_oshift+200_olen500

PATIENTS=(
  PTYEU_task147 PTYFF_task17 PTYFG_task18 PTYFI_task81
  PTYFA_task25 PTYFK_task40 PTYEY_task86 PTYEV_task37
  PTYEZ_task60 PTYFC_task28 PTYFM_task104 PTYFP_task88
  PTYFR_task91 PTYFS_task95 PTYFU_task224
)

patient_idx=$((SLURM_ARRAY_TASK_ID % 15))
layer=$((SLURM_ARRAY_TASK_ID / 15))
patient="${PATIENTS[$patient_idx]}"

"$PY" -u scripts/semantic_glm.py \
  --patient "$patient" \
  --region hippocampus \
  --model bert-base-causal \
  --context_tag _ctx200spktag \
  --layer "$layer" \
  --n_components 50 \
  --window_type fixed \
  --spike_tag "$TAG" \
  --notebook_exact \
  --outer_cv shuffle \
  --perm_type xcirc \
  --r2_only \
  --force_all
