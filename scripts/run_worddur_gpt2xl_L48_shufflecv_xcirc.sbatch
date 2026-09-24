#!/usr/bin/env bash
# CV-method robustness check (2 of 2): worddur window, GPT2-XL L48, pc=20,
# with REAL xcirc permutation nulls under --outer_cv shuffle instead of
# layer_sweep.sbatch's --outer_cv block (which produced the existing
# gpt2-xl_ctx200_worddur_xcirc/pc20 data used in point 2's table, showing
# ~75-87% significant). Matches layer_sweep.sbatch's invocation exactly
# (no --spike_tag override, no --notebook_exact, no --force_all) except
# --outer_cv and --n_perm (100 here vs 50 there, matching today's convention),
# so the only real difference from the existing block-CV worddur result is
# the CV-splitting method.

#SBATCH --job-name=worddur_gpt2xl_L48_shufcv
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/cv_method_check/worddur_shufcv_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/cv_method_check/worddur_shufcv_%j.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${PROJECT}"
mkdir -p logs/cv_method_check

echo "model=gpt2-xl layer=48 pc=20 (worddur window, SHUFFLE cv, xcirc null, n_perm=100)"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model gpt2-xl \
    --context_tag _ctx200 \
    --layer 48 \
    --n_components 20 \
    --region hippocampus \
    --window_type worddur \
    --outer_cv shuffle \
    --perm_type xcirc \
    --n_perm 100

echo "completed=$(date --iso-8601=seconds)"
