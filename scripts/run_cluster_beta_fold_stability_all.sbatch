#!/usr/bin/env bash
#SBATCH --job-name=beta_stab_all
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-22%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/beta_stability_all_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/beta_stability_all_%A_%a.err

set -euo pipefail

# PTYEZ_task60 hippocampus is omitted because its validated fast result exists.
JOBS=(
  "PTYEU_task147 hippocampus"
  "PTYEU_task147 ACC"
  "PTYFF_task17 hippocampus"
  "PTYFF_task17 ACC"
  "PTYFG_task18 hippocampus"
  "PTYFG_task18 ACC"
  "PTYFI_task81 hippocampus"
  "PTYFI_task81 ACC"
  "PTYFA_task25 hippocampus"
  "PTYFA_task25 ACC"
  "PTYFK_task40 hippocampus"
  "PTYFK_task40 ACC"
  "PTYEY_task86 hippocampus"
  "PTYEV_task37 hippocampus"
  "PTYEV_task37 ACC"
  "PTYEZ_task60 ACC"
  "PTYFC_task28 hippocampus"
  "PTYFC_task28 ACC"
  "PTYFM_task104 hippocampus"
  "PTYFP_task88 hippocampus"
  "PTYFR_task91 hippocampus"
  "PTYFS_task95 hippocampus"
  "PTYFU_task224 hippocampus"
)

read -r patient region <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
cd /scratch/aniluchavez/hippocampal-speaker-semantics
/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/cluster_beta_fold_stability.py \
  --patient "$patient" --region "$region" --folds 5 --fast-alpha
