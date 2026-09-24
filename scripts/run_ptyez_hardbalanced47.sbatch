#!/usr/bin/env bash
#SBATCH --job-name=ptyez_bal47
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/ptyez_bal47_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/ptyez_bal47_%j.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/cluster_glm_worddur_offset.py \
  --patient PTYEZ_task60 \
  --region hippocampus \
  --exclude-clusters 2 4 7 \
  --hard-balance \
  --results-root /scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise_worddur_offset_hardbalanced47_no_mental_places_visual
