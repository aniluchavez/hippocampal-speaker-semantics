#!/usr/bin/env bash
# Completes the one missing patient (PTYFU_task224) for the
# fixed_selfm300_otherp200_len500 config from run_rcross_window_sweep.sbatch
# (array task 7 died partway through with no error, ~37 min in, well under
# the 4hr budget -- looked like an external kill, not a code issue).

#SBATCH --job-name=rcross_window_finish
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/rcross_window_sweep/finish_last_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/rcross_window_sweep/finish_last_%j.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${PROJECT}"
mkdir -p logs/rcross_window_sweep

echo "host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model gpt2-xl \
    --context_tag _ctx200 \
    --layer 48 \
    --n_components 100 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag fixed_selfm300_otherp200_len500 \
    --outer_cv shuffle \
    --force_all \
    --notebook_exact \
    --r2_only \
    --perm_type xcirc \
    --reliability \
    --rel_n_null 50 \
    --rel_n_half_splits 50 \
    --rel_n_jobs 12 \
    --patient PTYFU_task224

echo "completed=$(date --iso-8601=seconds)"
