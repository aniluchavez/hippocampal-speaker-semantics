#!/usr/bin/env bash
#SBATCH --job-name=beta_stab
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/beta_stability_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/beta_stability_%j.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/cluster_beta_fold_stability.py \
  --patient PTYEZ_task60 --region hippocampus --folds 5 --fast-alpha
