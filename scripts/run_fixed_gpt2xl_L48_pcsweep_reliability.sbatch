#!/usr/bin/env bash
# Same config as run_fixed_gpt2xl_L48_pcsweep.sbatch (gpt2-xl, last layer 48,
# fixed window self=[onset-200,+300] other=[onset+100,+600], 5-fold shuffle
# outer CV) but with --reliability instead of the xcirc/xshuffle R2
# permutation null. This computes, per neuron: r_cross = corr(beta_self,
# beta_other) + a permutation null for it, plus within-condition split-half
# beta reliability -> cross-condition noise ceiling sqrt(SB(r_self)*SB(r_other))
# (neural_encoding/reliability.py, matching BERTRidgeBetaCorr.ipynb).
#
# --r2_only skips the (already-computed elsewhere) R2 permutation null so
# compute is spent entirely on the beta-reliability analysis.
# --rel_n_null / --rel_n_half_splits reduced to 50 (from default 100) to
# bound runtime -- these are CPU-bound sklearn PoissonRegressor fits, not
# GPU-batched, and cost grows with PC count (more features per fit).
#
# Run across the full PC sweep to check whether the self/other beta
# correlation (the geometric/subspace claim) is robust to PC count choice.
#
# Output lands in a *new* directory (perm_suffix picks up r2_only + xcirc,
# same as the earlier r2_only runs, but now also carries "reliability" data
# inside each pkl's "reliability" key):
#   gpt2-xl_ctx200_fixed_selfm200_otherp100_len500_notebookexact_shuf_xcirc_r2only/pc{N}/
# NOTE: this is the SAME directory as the original (no-reliability) r2_only
# sweep -- the pkls there lack a "reliability" key, so semantic_glm.py's
# cache check (line ~702) will detect that and recompute/overwrite them with
# the reliability-augmented version. This does not touch the separate
# _xcirc / _xshuffle (non-r2only) null-check directories.
#
# Array index -> PC count:
#   0:5  1:10  2:20  3:50  4:100  5:200  6:300  7:500

#SBATCH --job-name=fixed_gpt2xl_L48_reliability
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48_reliability/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48_reliability/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
SPIKE_TAG="fixed_selfm200_otherp100_len500"

pc_counts=(5 10 20 50 100 200 300 500)
i="${SLURM_ARRAY_TASK_ID}"
pc="${pc_counts[$i]}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${PROJECT}"
mkdir -p logs/fixed_gpt2xl_L48_reliability

echo "model=gpt2-xl layer=48 pc=${pc} (self/other beta reliability, n_null=50, n_half_splits=50)"
echo "window self=[onset-200, onset+300], other=[onset+100, onset+600]"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model gpt2-xl \
    --context_tag _ctx200 \
    --layer 48 \
    --n_components "${pc}" \
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
    --rel_n_jobs 12

echo "completed=$(date --iso-8601=seconds)"
