#!/usr/bin/env bash
#SBATCH --job-name=w2v_shift10
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-9%5
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/w2v_shift10_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/w2v_shift10_%A_%a.err

set -euo pipefail

cd /scratch/aniluchavez/hippocampal-speaker-semantics
mkdir -p logs

PATIENTS=(
  PTYFC_task28
  PTYFI_task81
  PTYEU_task147
  PTYFG_task18
  PTYEZ_task60
  PTYFA_task25
  PTYEV_task37
  PTYFK_task40
  PTYFF_task17
  PTYEY_task86
)

patient="${PATIENTS[$SLURM_ARRAY_TASK_ID]}"
echo "Running ${patient} on ${SLURM_CPUS_PER_TASK} CPUs"

/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/runningthroughwindows_fasttext_shift_sweep.py \
  --patient "${patient}" \
  --region hippocampus \
  --shift-start -1000 \
  --shift-stop 1000 \
  --shift-step 10 \
  --test-size none \
  --n-pcs 100 \
  --n-shuffles 20 \
  --n-jobs "${SLURM_CPUS_PER_TASK}"
