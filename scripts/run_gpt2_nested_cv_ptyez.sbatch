#!/usr/bin/env bash
#SBATCH --job-name=gpt2_nested_y
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/gpt2_nested_ptyez_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/gpt2_nested_ptyez_%j.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/gpt2_fixed_nested_predictive_cv.py \
  --patient PTYEZ_task60 \
  --n-components 10 \
  --extended-alpha \
  --results-root /scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise_gpt2large_L36_fixed_custom_softbalanced
