#!/usr/bin/env bash
# Self/other (speaking vs listening) BERT L12 pc100 reliability, matching
# run_selfother_reliability_nnull1000_parallel.sbatch (pc50) but at
# n_components=100, to check whether the manuscript's stated "100 PCs"
# reproduces at that PC count instead of the pc50 data actually on disk.
# rel_n_null=1000 for a fine permutation-null p-value floor (~0.0009).
# One array task per (patient, model) pair for maximum parallelism across
# both GPU partitions.

#SBATCH --job-name=selfother_rel_pc100
#SBATCH --partition=guppy,mammoth
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=0-29%8
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/selfother_rel_pc100/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/selfother_rel_pc100/%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
mkdir -p logs/selfother_rel_pc100

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
SPIKE_TAG="bestfixed_selfm300_len500_otherp20_len500"

PATIENTS=(
  PTYEU_task147
  PTYEV_task37
  PTYEY_task86
  PTYEZ_task60
  PTYFA_task25
  PTYFC_task28
  PTYFF_task17
  PTYFG_task18
  PTYFI_task81
  PTYFK_task40
  PTYFM_task104
  PTYFP_task88
  PTYFR_task91
  PTYFS_task95
  PTYFU_task224
)
N_PATIENTS=${#PATIENTS[@]}

model_idx=$(( SLURM_ARRAY_TASK_ID / N_PATIENTS ))
patient_idx=$(( SLURM_ARRAY_TASK_ID % N_PATIENTS ))
PATIENT="${PATIENTS[$patient_idx]}"

echo "task=${SLURM_ARRAY_TASK_ID} patient=${PATIENT} model_idx=${model_idx} host=$(hostname) started=$(date --iso-8601=seconds)"

if [[ "${model_idx}" == "0" ]]; then
    echo "--- BERT no-tag ---"
    "${PYTHON}" -u scripts/semantic_glm.py \
        --model bert-base \
        --context_tag _ctx200 \
        --layer 12 \
        --n_components 100 \
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
else
    echo "--- BERT spktag (causal) ---"
    "${PYTHON}" -u scripts/semantic_glm.py \
        --model bert-base-causal \
        --context_tag _ctx200spktag \
        --layer 12 \
        --n_components 100 \
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
fi

echo "completed=$(date --iso-8601=seconds)"
