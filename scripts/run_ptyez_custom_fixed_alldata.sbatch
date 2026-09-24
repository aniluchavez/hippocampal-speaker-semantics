#!/usr/bin/env bash
#SBATCH --job-name=ptyez_all
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/ptyez_all_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/ptyez_all_%j.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
TAG=tref-offset_shift-200_tlen500_oref-onset_shift+100_olen500
/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u \
  scripts/cluster_glm_legacy_replication.py \
  --patient PTYEZ_task60 \
  --region hippocampus \
  --n_jobs 8 \
  --window-tag "$TAG" \
  --no-balance \
  --results-root /scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise_fixed_self_offsetm200_other_onsetp100_alldata
