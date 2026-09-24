#!/usr/bin/env bash
#SBATCH --job-name=beta_curve_y
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/beta_curve_y_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/beta_curve_y_%j.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/beta_variance_learning_curve.py \
  --patient PTYEZ_task60 \
  --repeats 25 \
  --n-jobs 8 \
  --results-root /scratch/aniluchavez/ConvoDATAS/SemanticGLM/beta_variance_learning_curves_gpt2large_L36
