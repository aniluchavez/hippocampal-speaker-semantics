#!/usr/bin/env bash
# Follow-up for the 4 spktag patients whose mammoth-running tasks were
# cancelled (561000_15,19,20,21) when consolidating onto guppy only.
#SBATCH --job-name=selfother_rel_nnull1000_missing
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/selfother_rel_nnull1000/missing_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/selfother_rel_nnull1000/missing_%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
SPIKE_TAG="bestfixed_selfm300_len500_otherp20_len500"

PATIENTS=(PTYEU_task147 PTYFA_task25 PTYFC_task28 PTYFF_task17)
PATIENT="${PATIENTS[$SLURM_ARRAY_TASK_ID]}"

echo "patient=${PATIENT} host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model bert-base-causal \
    --context_tag _ctx200spktag \
    --layer 12 \
    --n_components 50 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag "${SPIKE_TAG}" \
    --outer_cv shuffle \
    --force_all \
    --notebook_exact \
    --r2_only \
    --reliability \
    --rel_n_null 1000 \
    --rel_n_half_splits 100 \
    --rel_n_jobs 8 \
    --patient "${PATIENT}"

echo "completed=$(date --iso-8601=seconds)"
