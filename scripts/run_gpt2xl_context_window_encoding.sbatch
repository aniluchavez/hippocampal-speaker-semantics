#!/usr/bin/env bash
#SBATCH --job-name=gpt2xl_ctx_glm
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-6
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/gpt2xl_context_glm/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/gpt2xl_context_glm/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
SPIKE_TAG="bestfixed_selfm300_len500_otherp20_len500"

# Must match run_gpt2xl_context_embeddings.sbatch.
contexts=(0 25 50 100 200 400 800)
ctx="${contexts[$SLURM_ARRAY_TASK_ID]}"
context_tag="_ctx${ctx}"

mkdir -p "${PROJECT}/logs/gpt2xl_context_glm"
cd "${PROJECT}"

echo "host=$(hostname)"
echo "context_words=${ctx} context_tag=${context_tag} model=gpt2-xl layer=48 pc=50 spike_tag=${SPIKE_TAG}"
echo "started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model gpt2-xl \
    --context_tag "${context_tag}" \
    --layer 48 \
    --n_components 50 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag "${SPIKE_TAG}" \
    --outer_cv shuffle \
    --force_all \
    --notebook_exact \
    --r2_only

echo "completed=$(date --iso-8601=seconds)"
