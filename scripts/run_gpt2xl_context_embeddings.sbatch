#!/usr/bin/env bash
#SBATCH --job-name=gpt2xl_ctx_embed
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-6
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/gpt2xl_context_embeddings/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/gpt2xl_context_embeddings/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"

# 0 = current word only; 200 matches the existing analysis/caches.
contexts=(0 25 50 100 200 400 800)
ctx="${contexts[$SLURM_ARRAY_TASK_ID]}"
context_tag="_ctx${ctx}"

mkdir -p "${PROJECT}/logs/gpt2xl_context_embeddings"
cd "${PROJECT}"

echo "host=$(hostname)"
echo "context_words=${ctx} context_tag=${context_tag} model=gpt2-xl"
echo "started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/extract_embeddings_ctx200.py \
    --models gpt2-xl \
    --context_words "${ctx}" \
    --context_tag "${context_tag}" \
    --max_tokens 1024

echo "completed=$(date --iso-8601=seconds)"
