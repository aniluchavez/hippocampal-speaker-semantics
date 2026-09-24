#!/usr/bin/env bash
#SBATCH --job-name=duration_only_check
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/duration_only_check/%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/duration_only_check/%j.err

set -euo pipefail
PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
export TRANSFORMERS_OFFLINE=1

cd "${PROJECT}"
mkdir -p logs/duration_only_check
echo "host=$(hostname) started=$(date --iso-8601=seconds)"
"${PYTHON}" -u scripts/duration_only_worddur_check.py
echo "completed=$(date --iso-8601=seconds)"
