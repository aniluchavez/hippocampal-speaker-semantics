#!/usr/bin/env bash
# PC-count sweep for GPT-2 XL, LAST transformer layer (48), at the variable
# windows used in run_varwin_reliability_models.sbatch:
#   self  = onset - 300 ms to natural word offset
#   other = word onset to natural word offset + 100 ms
#
# Replicates the train/test R^2 vs. n_components curve from
# notebooks/01_pc_robustness.ipynb (figures/01_train_test_curve.pdf), which
# was originally built for gpt2-xl_ctx200_worddur_xshuffle L36 — this run
# targets the new varwin window instead, at layer 48 (the architectural last
# layer, confirmed via EmbedCache shape (49, n_words, 1600) -> 48 transformer
# layers + embedding layer 0).
#
# --reliability is intentionally omitted (the existing pc50/L45 run used it
# and took ~50 min/patient-set); OUT_DIR naming does not depend on that flag
# (semantic_glm.py:159-176), so this writes into the same directory tree.
#
# Array index -> PC count:
#   0:5  1:10  2:20  3:50  4:100  5:200  6:300  7:500

#SBATCH --job-name=varwin_gpt2xl_L48
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/varwin_gpt2xl_L48/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/varwin_gpt2xl_L48/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
SPIKE_TAG="varwin_selfm300tooffset0_other0tooffsetp100"

pc_counts=(5 10 20 50 100 200 300 500)
i="${SLURM_ARRAY_TASK_ID}"
pc="${pc_counts[$i]}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${PROJECT}"
mkdir -p logs/varwin_gpt2xl_L48

echo "model=gpt2-xl layer=48 pc=${pc}"
echo "window self=[onset-300, offset], other=[onset, offset+100]"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model gpt2-xl \
    --context_tag _ctx200 \
    --layer 48 \
    --n_components "${pc}" \
    --region hippocampus \
    --window_type varwin \
    --spike_tag "${SPIKE_TAG}" \
    --outer_cv shuffle \
    --force_all \
    --notebook_exact \
    --r2_only \
    --perm_type xcirc

echo "completed=$(date --iso-8601=seconds)"
