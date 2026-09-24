#!/usr/bin/env bash
# Rerun of run_bert_L12_rcross_window_length_sweep.sbatch WITHOUT --r2_only, so
# the real block-shuffle (xcirc) permutation significance test runs (n_perm=500,
# no refit) in addition to reliability/r_cross. The earlier r2_only run left
# "significant" forced to 0/N for every neuron — this regenerates the same
# pkls with real p-values so a %-significant-by-length table can be built.
# Same placement (self=-300ms, other=+20ms), all 5 lengths incl. 500 (which
# was also only ever run with r2_only, in the placement sweep).

#SBATCH --job-name=bert_rcross_len_sig
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_rcross_window_length_sweep_sig/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_rcross_window_length_sweep_sig/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"

lengths=(200 300 500 700 800)

SELF_START=-300
OTHER_START=20

i="${SLURM_ARRAY_TASK_ID}"
LENGTH="${lengths[$i]}"
TAG="fixed_selfm300_otherp20_len${LENGTH}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${PROJECT}"
mkdir -p logs/bert_rcross_window_length_sweep_sig

echo "config idx=${i} tag=${TAG} self_start=${SELF_START} other_start=${OTHER_START} length=${LENGTH}"
echo "model=bert-base context=_ctx200 layer=12 pc=100 (real xcirc perm test, n_perm=500)"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

echo "--- generating spike windows for all 15 patients (idempotent) ---"
"${PYTHON}" -u scripts/generate_spike_windows.py \
    --mode fixed \
    --target_shift "${SELF_START}" \
    --other_shift "${OTHER_START}" \
    --target_window_length "${LENGTH}" \
    --other_window_length "${LENGTH}" \
    --out_tag "${TAG}" \
    --clip_bounds

echo "--- running semantic_glm.py (full significance test + reliability) ---"
"${PYTHON}" -u scripts/semantic_glm.py \
    --model bert-base \
    --context_tag _ctx200 \
    --layer 12 \
    --n_components 100 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag "${TAG}" \
    --outer_cv shuffle \
    --force_all \
    --notebook_exact \
    --perm_type xcirc \
    --n_perm 500 \
    --reliability \
    --rel_n_null 50 \
    --rel_n_half_splits 50 \
    --rel_n_jobs 12

echo "completed=$(date --iso-8601=seconds)"
