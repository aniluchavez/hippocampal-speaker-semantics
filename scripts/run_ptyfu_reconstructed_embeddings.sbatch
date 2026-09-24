#!/bin/bash
# Regenerate ctx200 embeddings for PTYFU_task224 only, against the newly
# reconstructed transcript (word count changed 12355 -> 12394, so the old
# cache is stale and misaligned). Scoped to the 9 models geometry_paper's
# CONVO14 pipeline uses (skips gpt2-large/gpt2-xl/bert-base-causal).
#
# Submit with: sbatch scripts/run_ptyfu_reconstructed_embeddings.sbatch

#SBATCH --job-name=ptyfu_emb
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/ptyfu_emb_%j.out
#SBATCH --partition=guppy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
mkdir -p logs

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
$PY -u scripts/extract_embeddings_ctx200.py \
    --patients PTYFU_task224 \
    --models gpt2,gpt2-medium,bert-base,roberta-base,opt-350m,llama-2-7b,llama-3.1-8b,gemma-2-9b,mistral-7b \
    --context_words 200 --max_tokens 512 --overwrite
