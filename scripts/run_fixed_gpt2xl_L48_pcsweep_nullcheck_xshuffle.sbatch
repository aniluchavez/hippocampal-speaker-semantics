#!/usr/bin/env bash
# Same as run_fixed_gpt2xl_L48_pcsweep_nullcheck.sbatch but with a plain
# global random permutation of word order (--perm_type xshuffle) instead of
# the lag-constrained circular shift (xcirc). Full random shuffle destroys
# any shared slow drift between embeddings and neural activity along with
# the word-by-word alignment, unlike xcirc which only breaks the alignment.
# n_perm=100, same as the xcirc nullcheck run, for a like-for-like comparison.
#
# Output lands in a *new* directory (perm_type name is baked into the path):
#   gpt2-xl_ctx200_fixed_selfm200_otherp100_len500_notebookexact_shuf_xshuffle/pc{N}/
#
# Array index -> PC count:
#   0:5  1:10  2:20  3:50  4:100  5:200  6:300  7:500

#SBATCH --job-name=fixed_gpt2xl_L48_nullcheck_xshuf
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48_nullcheck_xshuffle/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48_nullcheck_xshuffle/%A_%a.err

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
mkdir -p logs/fixed_gpt2xl_L48_nullcheck_xshuffle

echo "model=gpt2-xl layer=48 pc=${pc} (xshuffle null, n_perm=100)"
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
    --perm_type xshuffle \
    --n_perm 100

echo "completed=$(date --iso-8601=seconds)"
