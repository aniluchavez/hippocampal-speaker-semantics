#!/usr/bin/env bash
#SBATCH --job-name=redboot_yey
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/redboot_yey_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/redboot_yey_%j.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/joint_reduced_model_bootstrap.py \
  --patient PTYEZ_task60 \
  --n-bootstraps 1000 \
  --n-jobs 8 \
  --n-components 10 \
  --deviation-penalty-multiplier 10 \
  --results-root /scratch/aniluchavez/ConvoDATAS/SemanticGLM/joint_category_interaction_gpt2large_L36
