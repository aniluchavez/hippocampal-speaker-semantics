#!/usr/bin/env bash
# BERT L12 reliability PC sweep for canonical fixed windows:
#   self  = onset - 200 ms to onset + 300 ms
#   other = onset + 200 ms to onset + 700 ms
#
# Default is raw/unbalanced trial counts ("real numbers").
# To run the balanced version instead:
#   sbatch --export=ALL,BALANCE_TRIALS=1 scripts/run_bert_l12_fixed_reliability_pc_sweep.sbatch

#SBATCH --job-name=bert_rel_pcs
#SBATCH --partition=mammoth
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_l12_fixed_reliability_pc_sweep/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_l12_fixed_reliability_pc_sweep/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
SPIKE_TAG="fixed_selfm200_otherp200_len500"
BALANCE_TRIALS="${BALANCE_TRIALS:-0}"

pcs=(5 10 20 50 100 200 300 500)
pc="${pcs[$SLURM_ARRAY_TASK_ID]}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${PROJECT}"
mkdir -p logs/bert_l12_fixed_reliability_pc_sweep

echo "model=bert-base_ctx200 layer=12 PCs=${pc}"
echo "window self=[onset-200,onset+300], other=[onset+200,onset+700], length=500"
echo "reliability BALANCE_TRIALS=${BALANCE_TRIALS}"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

echo "--- generating/checking fixed spike windows for all 15 patients ---"
"${PYTHON}" -u scripts/generate_spike_windows.py \
    --mode fixed \
    --target_shift -200 \
    --other_shift 200 \
    --target_window_length 500 \
    --other_window_length 500 \
    --out_tag "${SPIKE_TAG}" \
    --clip_bounds

args=()
if [[ "${BALANCE_TRIALS}" == "1" ]]; then
    args+=(--rel_balance_trials --rel_balance_seed 20260709)
fi

echo "--- running semantic_glm.py --reliability for pc=${pc} ---"
"${PYTHON}" -u scripts/semantic_glm.py \
    --model bert-base \
    --context_tag _ctx200 \
    --layer 12 \
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
    --rel_n_null 100 \
    --rel_n_half_splits 100 \
    --rel_n_jobs 12 \
    "${args[@]}"

echo "completed=$(date --iso-8601=seconds)"
