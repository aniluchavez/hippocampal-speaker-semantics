#!/usr/bin/env bash
# Completes the null-type check for worddur windows: same config as
# run_worddur_gpt2xl_L48_shufflecv_xcirc.sbatch (worddur, GPT2-XL L48, pc=20,
# outer_cv=shuffle) but --perm_type xshuffle instead of xcirc, to check
# whether the naive full-shuffle null is also miscalibrated here the way it
# was for fixed windows (76.6-91.5% vs xcirc's 17.1-27.0% at matching pc=20),
# or whether worddur behaves differently under xshuffle specifically.

#SBATCH --job-name=worddur_gpt2xl_L48_xshuf
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/cv_method_check/worddur_shufcv_xshuffle_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/cv_method_check/worddur_shufcv_xshuffle_%j.err

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

echo "model=gpt2-xl layer=48 pc=20 (worddur window, SHUFFLE cv, XSHUFFLE null, n_perm=100)"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model gpt2-xl \
    --context_tag _ctx200 \
    --layer 48 \
    --n_components 20 \
    --region hippocampus \
    --window_type worddur \
    --outer_cv shuffle \
    --perm_type xshuffle \
    --n_perm 100

echo "completed=$(date --iso-8601=seconds)"
