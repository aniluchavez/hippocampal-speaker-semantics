#!/usr/bin/env bash
# Balanced self/other reliability run at the requested canonical fixed window:
#   self  = onset - 200 ms to onset + 300 ms
#   other = onset + 200 ms to onset + 700 ms
#
# The main CV fit still uses all available words.  The --rel_balance_trials
# flag downsamples only the reliability inputs so self/other have matched
# trial counts per patient/region before r_cross and split-half ceilings.

#SBATCH --job-name=fixed_rel_bal30
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_reliability_balanced_pc30/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_reliability_balanced_pc30/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
SPIKE_TAG="fixed_selfm200_otherp200_len500"

models=(bert-base bert-base-causal llama-3.1-8b gpt2-xl)
contexts=(_ctx200 _ctx200spktag _ctx200 _ctx200)
layers=(12 12 14 45)
labels=(bert bert-speaker-tag llama31 gpt2-xl)

i="${SLURM_ARRAY_TASK_ID}"
model="${models[$i]}"
context="${contexts[$i]}"
layer="${layers[$i]}"
label="${labels[$i]}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${PROJECT}"
mkdir -p logs/fixed_reliability_balanced_pc30

echo "model=${model}${context} layer=${layer} label=${label}"
echo "window self=[onset-200, onset+300], other=[onset+200, onset+700], length=500"
echo "PCs=30 reliability balanced self/other trials"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

echo "--- generating fixed spike windows for all 15 patients if needed ---"
"${PYTHON}" -u scripts/generate_spike_windows.py \
    --mode fixed \
    --target_shift -200 \
    --other_shift 200 \
    --target_window_length 500 \
    --other_window_length 500 \
    --out_tag "${SPIKE_TAG}" \
    --clip_bounds

echo "--- running semantic_glm.py --reliability --rel_balance_trials ---"
"${PYTHON}" -u scripts/semantic_glm.py \
    --model "${model}" \
    --context_tag "${context}" \
    --layer "${layer}" \
    --n_components 30 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag "${SPIKE_TAG}" \
    --outer_cv shuffle \
    --force_all \
    --notebook_exact \
    --r2_only \
    --perm_type xcirc \
    --reliability \
    --rel_balance_trials \
    --rel_balance_seed 20260709 \
    --rel_n_null 100 \
    --rel_n_half_splits 100 \
    --rel_n_jobs 12

echo "completed=$(date --iso-8601=seconds)"
