#!/usr/bin/env bash
# BERT L12, fixed window, pc=30, outer_cv=BLOCK (default), xshuffle null.
# Same window/spike_tag as the pc20 shuffle-CV BERT table cells, just at
# pc=30 with block CV instead of shuffle CV.

#SBATCH --job-name=fixed_bert_L12_pc30_blockcv_xshuf
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/cv_method_check/fixed_bert_L12_pc30_blockcv_xshuffle_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/cv_method_check/fixed_bert_L12_pc30_blockcv_xshuffle_%j.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
SPIKE_TAG="fixed_selfm200_otherp100_len500"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${PROJECT}"
mkdir -p logs/cv_method_check

echo "model=bert-base layer=12 pc=30 (fixed window, BLOCK cv, xshuffle null, n_perm=100)"
echo "window self=[onset-200, onset+300], other=[onset+100, onset+600]"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model bert-base \
    --context_tag _ctx200 \
    --layer 12 \
    --n_components 30 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag "${SPIKE_TAG}" \
    --outer_cv block \
    --force_all \
    --notebook_exact \
    --perm_type xshuffle \
    --n_perm 100

echo "completed=$(date --iso-8601=seconds)"
