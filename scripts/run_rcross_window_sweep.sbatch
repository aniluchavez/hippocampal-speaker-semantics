#!/usr/bin/env bash
# r_cross-across-window-shifts sweep, GPT2-XL L48, pc=100, hippocampus.
# 8 new fixed-window configs (length=500ms, varying self/other onset-relative
# starts) + the session's existing window (self=-200,other=+100, already
# covered by fixed_gpt2xl_L48_pcsweep_rcross_table.tex at pc=100 -- not
# rerun here). Points chosen to span the range found in the existing 72-
# config gpt2-large L36 pc50 R2-only discovery sweep (best/near-best/worst-
# ish/contrasting-extreme starts), now checked for r_cross specifically on
# gpt2-xl L48 (this session's primary model), matching the "bestfixed"
# canonical window (self=-300,other=+20) used in point1's fixed-vs-variable
# r_cross table for continuity.
#
# Each array task: generate spike windows for all 15 patients for one config
# (generate_spike_windows.py defaults to all patients when --patients is
# omitted), then run semantic_glm.py --reliability for that config.

#SBATCH --job-name=rcross_window_sweep
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%3
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/rcross_window_sweep/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/rcross_window_sweep/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"

# index: 0        1        2        3        4        5        6        7
tags=(
  "fixed_selfm300_otherp20_len500"
  "fixed_selfm500_otherp20_len500"
  "fixed_selfm500_otherp200_len500"
  "fixed_selfm400_other0_len500"
  "fixed_selfm100_other0_len500"
  "fixed_self0_other0_len500"
  "fixed_self0_otherp200_len500"
  "fixed_selfm300_otherp200_len500"
)
self_starts=(-300 -500 -500 -400 -100 0    0    -300)
other_starts=(20   20   200  0    0    0    200  200)

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
mkdir -p logs/rcross_window_sweep

echo "config idx=${i} tag=${TAG} self_start=${SELF_START} other_start=${OTHER_START} length=${LENGTH}"
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
    --model gpt2-xl \
    --context_tag _ctx200 \
    --layer 48 \
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
