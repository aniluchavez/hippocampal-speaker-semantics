#!/usr/bin/env bash
#SBATCH --job-name=bertmax_spk_l12
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bertmax_spktag_bestfixed_L12_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bertmax_spktag_bestfixed_L12_%A_%a.err

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

patient="${PATIENTS[$SLURM_ARRAY_TASK_ID]}"

"$PY" -u scripts/semantic_glm.py \
  --patient "$patient" \
  --region hippocampus \
  --model bert-base-bidirmax-spktag \
  --context_tag "" \
  --layer 12 \
  --n_components 50 \
  --window_type fixed \
  --spike_tag "$TAG" \
  --notebook_exact \
  --outer_cv shuffle \
  --perm_type xcirc \
  --r2_only \
  --force_all
