#!/usr/bin/env bash
# PC-count sweep for GPT-2 XL, LAST transformer layer (48), at a FIXED window:
#   self  = onset - 200 ms, length 500 ms   (i.e. [onset-200, onset+300])
#   other = onset + 100 ms, length 500 ms   (i.e. [onset+100, onset+600])
#
# Companion to run_varwin_gpt2xl_L48_pcsweep.sbatch (same model/layer/PC sweep/
# CV scheme), just swapping the varwin window for this fixed-length window, to
# regenerate the 01_train_test_curve-style plot under fixed instead of variable
# windows. window_type=fixed means no Poisson exposure offset is applied (both
# self/other windows are constant length, so no duration normalization needed;
# semantic_glm.py:114 USE_OFFSET = WINDOW_TYPE in ("varwin","worddur") -> False
# here).
#
# Spike windows for this exact tag already exist for all 15 patients
# (output_pt*_english_only_fixed_selfm200_otherp100_len500); only the GLM fits
# are missing for gpt2-xl (gpt2-large already has this combo).
#
# Array index -> PC count:
#   0:5  1:10  2:20  3:50  4:100  5:200  6:300  7:500

#SBATCH --job-name=fixed_gpt2xl_L48
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48/%A_%a.err

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
mkdir -p logs/fixed_gpt2xl_L48

echo "model=gpt2-xl layer=48 pc=${pc}"
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
    --perm_type xcirc

echo "completed=$(date --iso-8601=seconds)"
