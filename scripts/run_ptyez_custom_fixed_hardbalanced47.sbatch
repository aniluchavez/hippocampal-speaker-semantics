#!/usr/bin/env bash
#SBATCH --job-name=ptyez_fix47
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/ptyez_fix47_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/ptyez_fix47_%j.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
TAG=tref-offset_shift-200_tlen500_oref-onset_shift+100_olen500

"$PY" -u scripts/generate_spike_windows.py \
  --mode fixed \
  --patients PTYEZ_task60 \
  --target_ref_point offset \
  --target_shift -200 \
  --target_window_length 500 \
  --other_shift 100 \
  --other_window_length 500 \
  --out_tag "$TAG"

"$PY" -u scripts/cluster_glm_legacy_replication.py \
  --patient PTYEZ_task60 \
  --region hippocampus \
  --n_jobs 8 \
  --window-tag "$TAG" \
  --exclude-clusters 2 4 7 \
  --hard-balance \
  --results-root /scratch/aniluchavez/ConvoDATAS/SemanticGLM/clusterwise_fixed_self_offsetm200_other_onsetp100_hardbalanced47
