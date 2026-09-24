#!/usr/bin/env bash
# BERT L12 fixed-vs-word-duration encoding significance with shuffled outer CV
# and shuffled-X null. This is intended to regenerate the fixed-vs-worddur
# comparison using BERT specifically, not the earlier circular-shift null.
#
# Array tasks:
#   0 = fixed 500 ms windows, self onset-200 ms, other onset+200 ms
#   1 = word-duration windows with Poisson exposure offset
#
# By default this uses 50 PCs, matching the current BERT reliability/control
# analyses. Override with:
#   sbatch --export=ALL,N_COMPONENTS=100 scripts/run_bert_l12_fixed_vs_worddur_xshuffle.sbatch

#SBATCH --job-name=bert_fw_wd_xshuf
#SBATCH --partition=mammoth
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=36:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_fixed_vs_worddur_xshuffle/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/bert_fixed_vs_worddur_xshuffle/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
N_COMPONENTS="${N_COMPONENTS:-50}"
SPIKE_TAG="fixed_selfm200_otherp200_len500"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${PROJECT}"
mkdir -p logs/bert_fixed_vs_worddur_xshuffle

echo "task=${SLURM_ARRAY_TASK_ID} host=$(hostname) started=$(date --iso-8601=seconds)"
echo "model=bert-base_ctx200 layer=12 PCs=${N_COMPONENTS} outer_cv=shuffle perm_type=xshuffle"

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
    echo "--- fixed window: self=[onset-200,onset+300], other=[onset+200,onset+700] ---"
    "${PYTHON}" -u scripts/generate_spike_windows.py \
        --mode fixed \
        --target_shift -200 \
        --other_shift 200 \
        --target_window_length 500 \
        --other_window_length 500 \
        --out_tag "${SPIKE_TAG}" \
        --clip_bounds

    "${PYTHON}" -u scripts/semantic_glm.py \
        --model bert-base \
        --context_tag _ctx200 \
        --layer 12 \
        --n_components "${N_COMPONENTS}" \
        --region hippocampus \
        --window_type fixed \
        --spike_tag "${SPIKE_TAG}" \
        --outer_cv shuffle \
        --force_all \
        --notebook_exact \
        --perm_type xshuffle \
        --n_perm 500
else
    echo "--- word-duration window with Poisson exposure offset ---"
    "${PYTHON}" -u scripts/semantic_glm.py \
        --model bert-base \
        --context_tag _ctx200 \
        --layer 12 \
        --n_components "${N_COMPONENTS}" \
        --region hippocampus \
        --window_type worddur \
        --outer_cv shuffle \
        --force_all \
        --notebook_exact \
        --perm_type xshuffle \
        --n_perm 500
fi

echo "completed=$(date --iso-8601=seconds)"
