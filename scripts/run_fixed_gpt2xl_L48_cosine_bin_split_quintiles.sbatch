#!/usr/bin/env bash
# Second follow-up to run_fixed_gpt2xl_L48_cosine_bin_split.sbatch's near/far
# median split and run_fixed_gpt2xl_L48_cosine_bin_split_terciles.sbatch's
# 3-bin follow-up: same reviewer question (is the self/other beta
# "semi-orthogonal" geometry just a context-overlap artifact?), now at
# quintile resolution (--cosine_n_bins 5) for a finer-grained look at whether
# the tercile result (near-tercile boost, flat mid/far) holds up or resolves
# into further structure at higher resolution.
#
# --cosine_min_words_per_bin defaults to 60 (3x n_components=20). Checked
# ahead of time from the tercile run's logged per-bin word counts: the same
# two patients that drop at tercile resolution (PTYEY_task86, PTYEV_task37 --
# both due to very small self-condition word counts, 67 and 113 words total)
# drop here too, and are the ONLY ones that drop -- every other patient's
# smallest quintile bin still clears 60 words. So n=13/15, same as terciles,
# no surprise attrition expected.
#
# pc=20, n_null=n_half_splits=50, matching the near/far and tercile runs for
# comparability.

#SBATCH --job-name=fixed_gpt2xl_L48_cosinebin_quintiles
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48_cosinebin/quintiles_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48_cosinebin/quintiles_%j.err

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
mkdir -p logs/fixed_gpt2xl_L48_cosinebin

echo "model=gpt2-xl layer=48 pc=20 (cosine-bin r_cross QUINTILE split, n_null=50, n_half_splits=50)"
echo "window self=[onset-200, onset+300], other=[onset+100, onset+600]"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model gpt2-xl \
    --context_tag _ctx200 \
    --layer 48 \
    --n_components 20 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag "${SPIKE_TAG}" \
    --outer_cv shuffle \
    --force_all \
    --notebook_exact \
    --r2_only \
    --perm_type xcirc \
    --reliability \
    --rel_n_null 50 \
    --rel_n_half_splits 50 \
    --rel_n_jobs 12 \
    --cosine_bin_split \
    --cosine_n_bins 5

echo "completed=$(date --iso-8601=seconds)"
