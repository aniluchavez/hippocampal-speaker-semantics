#!/usr/bin/env bash
# Same config as run_fixed_gpt2xl_L48_pcsweep.sbatch (gpt2-xl, last layer 48,
# fixed window self=[onset-200,+300] other=[onset+100,+600], 5-fold shuffle
# outer CV) but WITHOUT --r2_only, so the xcirc circular-shift permutation
# null is actually computed (100 perms instead of the project default 500,
# to keep wall time reasonable for an exploratory look). This lets
# 01_pc_robustness-style plotting add a "shuffled null" curve alongside the
# existing train/test R² curve, to check whether test R² clears chance.
#
# Output lands in a *new* directory (no _r2only suffix):
#   gpt2-xl_ctx200_fixed_selfm200_otherp100_len500_notebookexact_shuf_xcirc/pc{N}/
# — does not overwrite the existing r2_only results.
#
# Array index -> PC count:
#   0:5  1:10  2:20  3:50  4:100  5:200  6:300  7:500

#SBATCH --job-name=fixed_gpt2xl_L48_nullcheck
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48_nullcheck/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48_nullcheck/%A_%a.err

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
mkdir -p logs/fixed_gpt2xl_L48_nullcheck

echo "model=gpt2-xl layer=48 pc=${pc} (xcirc null, n_perm=100)"
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
    --perm_type xcirc \
    --n_perm 100

echo "completed=$(date --iso-8601=seconds)"
