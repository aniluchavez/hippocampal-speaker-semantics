#!/usr/bin/env bash
# Reviewer question: is the self/other beta-vector "semi-orthogonal" geometry
# (r_cross real but well below the noise ceiling) just an artifact of self and
# other words living in different conversational contexts?
#
# Tests this via a cosine-distance-to-other-centroid split (semantic_glm.py's
# new --cosine_bin_split, added this session): for each word, cosine distance
# to the OPPOSITE condition's raw-embedding centroid; median split into
# near/far bins per condition; rerun r_cross + noise ceiling separately on
# each bin. If the alignment survives in the "far" bin (words whose context
# looks nothing like the opposite condition's overall content), that argues
# against the context-overlap-artifact explanation.
#
# pc=20 only (per user request) -- not a sweep. --r2_only skips the
# (already-computed) R2 permutation null; n_null/n_half_splits=50 matches the
# earlier reliability sweep for comparability.
#
# Output: SAME directory as the earlier reliability sweep
#   gpt2-xl_ctx200_fixed_selfm200_otherp100_len500_notebookexact_shuf_xcirc_r2only/pc20/
# pkls there already have "reliability" (full-word) from the earlier sweep;
# this run detects the missing "reliability_cosine_bin" key and recomputes
# (full-word reliability gets redone too, unavoidable -- region_cond_xy isn't
# cached separately). No separate backup needed since the earlier reliability
# job already validated recompute is deterministic/safe here, and this only
# adds a new key, never removes "reliability".

#SBATCH --job-name=fixed_gpt2xl_L48_cosinebin
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48_cosinebin/%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48_cosinebin/%j.err

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

echo "model=gpt2-xl layer=48 pc=20 (cosine-bin r_cross split, n_null=50, n_half_splits=50)"
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
    --cosine_bin_split

echo "completed=$(date --iso-8601=seconds)"
