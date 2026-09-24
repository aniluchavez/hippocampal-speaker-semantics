#!/usr/bin/env bash
#SBATCH --job-name=w2v5_chip
#SBATCH --partition=chipmunk
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/w2v5_chip_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/w2v5_chip_%A_%a.err

set -euo pipefail

cd /scratch/aniluchavez/hippocampal-speaker-semantics
mkdir -p logs

PATIENTS=(
  PTYFM_task104
  PTYFP_task88
  PTYFR_task91
  PTYFS_task95
  PTYFU_task224
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
