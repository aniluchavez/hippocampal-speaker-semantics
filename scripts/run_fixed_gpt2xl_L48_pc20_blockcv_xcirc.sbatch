#!/usr/bin/env bash
# Completes the matched 2x2 (window type x CV method) comparison at pc=20:
# fixed windows, GPT2-XL L48, pc=20, outer_cv=BLOCK, real xcirc perms.
# Everything else matches the existing fixed pc=20 shuffle-CV result exactly
# (same spike_tag/notebook_exact/force_all), so this isolates CV method at a
# PC count that matches the worddur comparison cells we already have.

#SBATCH --job-name=fixed_gpt2xl_L48_pc20_blockcv
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/cv_method_check/fixed_pc20_blockcv_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/cv_method_check/fixed_pc20_blockcv_%j.err

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

echo "model=gpt2-xl layer=48 pc=20 (fixed window, BLOCK cv, xcirc null, n_perm=100)"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model gpt2-xl \
    --context_tag _ctx200 \
    --layer 48 \
    --n_components 20 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag "${SPIKE_TAG}" \
    --outer_cv block \
    --force_all \
    --notebook_exact \
    --perm_type xcirc \
    --n_perm 100

echo "completed=$(date --iso-8601=seconds)"
