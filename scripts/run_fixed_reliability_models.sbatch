#!/usr/bin/env bash
# Reliability sweep at the selected fixed windows:
#   self  = onset - 300 ms to onset + 200 ms
#   other = onset + 20 ms to onset + 520 ms

#SBATCH --job-name=fixed_rel
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_reliability/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_reliability/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
SPIKE_TAG="bestfixed_selfm300_len500_otherp20_len500"

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

cd "${PROJECT}"
mkdir -p logs/fixed_reliability

echo "model=${model}${context} layer=${layer} label=${label}"
echo "window self=[onset-300, onset+200], other=[onset+20, onset+520]"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model "${model}" \
    --context_tag "${context}" \
    --layer "${layer}" \
    --n_components 50 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag "${SPIKE_TAG}" \
    --outer_cv shuffle \
    --force_all \
    --notebook_exact \
    --r2_only \
    --reliability \
    --rel_n_null 100 \
    --rel_n_half_splits 100 \
    --rel_n_jobs 8

echo "completed=$(date --iso-8601=seconds)"
