#!/usr/bin/env bash
# Quick smoke test for the new --cosine_bin_split code path in semantic_glm.py
# before committing to the full 15-patient run: one patient, n_null=2,
# n_half_splits=2. Just needs to complete without error and produce sane
# near/far split sizes + a "reliability_cosine_bin" key in the output pkl.
#
# Uses pc=19 (not the real pc=20 used by the full run) so this writes to an
# isolated throwaway output directory -- the full run's cache-check only
# looks for key presence, not null-iteration count, so reusing pc=20 here
# would leave PTYEZ_task60 stuck with this test's low-quality n_null=2
# results forever.

#SBATCH --job-name=cosinebin_smoketest
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48_cosinebin/smoketest_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48_cosinebin/smoketest_%j.err

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

echo "SMOKETEST: model=gpt2-xl layer=48 pc=19 (throwaway) tercile split, n_null=2, n_half_splits=2"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

for PATIENT in PTYEZ_task60 PTYEV_task37; do
    echo "=== testing patient ${PATIENT} ==="
    "${PYTHON}" -u scripts/semantic_glm.py \
        --model gpt2-xl \
        --context_tag _ctx200 \
        --layer 48 \
        --n_components 19 \
        --region hippocampus \
        --window_type fixed \
        --spike_tag "${SPIKE_TAG}" \
        --outer_cv shuffle \
        --force_all \
        --notebook_exact \
        --r2_only \
        --perm_type xcirc \
        --reliability \
        --rel_n_null 2 \
        --rel_n_half_splits 2 \
        --rel_n_jobs 4 \
        --cosine_bin_split \
        --cosine_n_bins 3 \
        --patient "${PATIENT}"
done

echo "completed=$(date --iso-8601=seconds)"
