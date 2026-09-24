#!/usr/bin/env bash
#SBATCH --job-name=gpt2_stab
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/gpt2_stability_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/gpt2_stability_%j.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/gpt2_fixed_beta_fold_stability.py
