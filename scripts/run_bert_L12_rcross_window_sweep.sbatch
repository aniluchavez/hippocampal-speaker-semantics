#!/usr/bin/env bash
# r_cross-across-window-shifts sweep, BERT-base L12, pc=100, hippocampus.
# Mirrors fixed_gpt2xl_L48_window_shift_rcross_table.tex but swaps in
# bert-base_ctx200 embeddings while keeping the fixed 500 ms window placements.

#SBATCH --job-name=bert_rcross_win
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_rcross_window_sweep/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_rcross_window_sweep/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"

# Matches the six rows in fixed_gpt2xl_L48_window_shift_rcross_table.tex.
tags=(
  "fixed_selfm200_otherp100_len500"
  "fixed_selfm300_otherp20_len500"
  "fixed_selfm100_other0_len500"
  "fixed_self0_other0_len500"
  "fixed_self0_otherp200_len500"
  "fixed_selfm300_otherp200_len500"
)
self_starts=(-200 -300 -100 0 0 -300)
other_starts=(100 20 0 0 200 200)

i="${SLURM_ARRAY_TASK_ID}"
TAG="${tags[$i]}"
SELF_START="${self_starts[$i]}"
OTHER_START="${other_starts[$i]}"
LENGTH=500

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${PROJECT}"
mkdir -p logs/bert_rcross_window_sweep

echo "config idx=${i} tag=${TAG} self_start=${SELF_START} other_start=${OTHER_START} length=${LENGTH}"
echo "model=bert-base context=_ctx200 layer=12 pc=100"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

echo "--- generating spike windows for all 15 patients ---"
"${PYTHON}" -u scripts/generate_spike_windows.py \
    --mode fixed \
    --target_shift "${SELF_START}" \
    --other_shift "${OTHER_START}" \
    --target_window_length "${LENGTH}" \
    --other_window_length "${LENGTH}" \
    --out_tag "${TAG}" \
    --clip_bounds

echo "--- running semantic_glm.py --reliability ---"
"${PYTHON}" -u scripts/semantic_glm.py \
    --model bert-base \
    --context_tag _ctx200 \
    --layer 12 \
    --n_components 100 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag "${TAG}" \
    --outer_cv shuffle \
    --force_all \
    --notebook_exact \
    --r2_only \
    --perm_type xcirc \
    --reliability \
    --rel_n_null 50 \
    --rel_n_half_splits 50 \
    --rel_n_jobs 12

echo "completed=$(date --iso-8601=seconds)"
