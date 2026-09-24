#!/usr/bin/env bash
#SBATCH --job-name=perm_yey
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/perm_ptyez_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/perm_ptyez_%j.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/cluster_full_pipeline_permutation.py \
  --patient PTYEZ_task60 \
  --n-permutations 100 \
  --n-jobs 8 \
  --results-root /scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise_gpt2large_L36_fixed_custom_alldata_permutation
