#!/usr/bin/env bash
#SBATCH --job-name=noisefloor_y
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/noisefloor_y_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/noisefloor_y_%j.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/joint_category_noise_floor.py \
  --patient PTYEZ_task60 \
  --n-splits 100 \
  --n-jobs 8 \
  --n-components 10 \
  --deviation-penalty-multiplier 10 \
  --joint-results-root /scratch/aniluchavez/ConvoDATAS/SemanticGLM/joint_category_interaction_gpt2large_L36
