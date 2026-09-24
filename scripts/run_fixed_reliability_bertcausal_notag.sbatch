#!/usr/bin/env bash
# Apples-to-apples tag ablation: causal BERT WITHOUT speaker tags, same fixed
# window/pc50/layer as the existing bert-base-causal_ctx200spktag run.
#   self  = onset - 300 ms to onset + 200 ms
#   other = onset + 20 ms to onset + 520 ms

#SBATCH --job-name=fixed_rel_causal_notag
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_reliability/%j_causal_notag.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_reliability/%j_causal_notag.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
SPIKE_TAG="bestfixed_selfm300_len500_otherp20_len500"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${PROJECT}"
mkdir -p logs/fixed_reliability

echo "model=bert-base-causal_ctx200 layer=12 label=bert-causal-no-tag"
echo "window self=[onset-300, onset+200], other=[onset+20, onset+520]"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model bert-base-causal \
    --context_tag _ctx200 \
    --layer 12 \
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
