#!/usr/bin/env bash
# CV-method robustness check (1 of 2): fixed windows, GPT2-XL L48, pc=100,
# with REAL xcirc permutation nulls under --outer_cv block (contiguous
# temporal folds) instead of this session's usual --outer_cv shuffle.
# Everything else matches run_fixed_gpt2xl_L48_pcsweep_nullcheck.sbatch's
# pc=100 config exactly, so the only thing that differs from the existing
# shuffle-CV pc=100 result is the CV-splitting method.

#SBATCH --job-name=fixed_gpt2xl_L48_pc100_blockcv
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/cv_method_check/fixed_blockcv_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/cv_method_check/fixed_blockcv_%j.err

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

echo "model=gpt2-xl layer=48 pc=100 (fixed window, BLOCK cv, xcirc null, n_perm=100)"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model gpt2-xl \
    --context_tag _ctx200 \
    --layer 48 \
    --n_components 100 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag "${SPIKE_TAG}" \
    --outer_cv block \
    --force_all \
    --notebook_exact \
    --perm_type xcirc \
    --n_perm 100

echo "completed=$(date --iso-8601=seconds)"
