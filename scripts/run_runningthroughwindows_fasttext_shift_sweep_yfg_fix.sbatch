#!/usr/bin/env bash
#SBATCH --job-name=w2v_yfgfix
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/w2v_yfgfix_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/w2v_yfgfix_%j.err

set -euo pipefail

cd /scratch/aniluchavez/hippocampal-speaker-semantics
mkdir -p logs

echo "Rerunning PTYFG_task18 with corrected hippocampus mapping on ${SLURM_CPUS_PER_TASK} CPUs"

/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/runningthroughwindows_fasttext_shift_sweep.py \
  --patient PTYFG_task18 \
  --region hippocampus \
  --shift-start -1000 \
  --shift-stop 1000 \
  --shift-step 10 \
  --test-size none \
  --n-pcs 100 \
  --n-shuffles 20 \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --overwrite-results
