#!/usr/bin/env bash
# r_cross-across-window-LENGTHS sweep, BERT-base L12, pc=100, hippocampus.
# Keeps the "bestfixed" placement (self_start=-300ms, other_start=+20ms) fixed
# and varies the shared window length instead. Mirrors
# run_bert_L12_rcross_window_sweep.sbatch, which instead swept placement at a
# fixed 500ms length; len500 for this placement already exists from that sweep
# (tag fixed_selfm300_otherp20_len500) so it is not regenerated here.

#SBATCH --job-name=bert_rcross_len
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_rcross_window_length_sweep/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_rcross_window_length_sweep/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"

lengths=(200 300 700 800)

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
mkdir -p logs/bert_rcross_window_length_sweep

echo "config idx=${i} tag=${TAG} self_start=${SELF_START} other_start=${OTHER_START} length=${LENGTH}"
echo "model=bert-base context=_ctx200 layer=12 pc=100"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

echo "--- generating spike windows for all 15 patients ---"
"${PYTHON}" -u scripts/generate_spike_windows.py \
    --mode fixed \
    --target_shift "${SELF_START}" \
    --other_shift "${OTHER_START}" \
    --target_window_length "${LENGTH}" \
    --other_window_length "${LENGTH}" \
    --out_tag "${TAG}" \
    --clip_bounds

echo "--- running semantic_glm.py --reliability ---"
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
    --r2_only \
    --perm_type xcirc \
    --reliability \
    --rel_n_null 50 \
    --rel_n_half_splits 50 \
    --rel_n_jobs 12

echo "completed=$(date --iso-8601=seconds)"
