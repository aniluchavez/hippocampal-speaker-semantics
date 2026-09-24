#!/usr/bin/env bash
# Other-other (external speaker vs external speaker) BERT pc50 reliability,
# rerun with --n-null 1000 (was 100) for a finer permutation-null p-value
# floor (1/1001~=0.0009 instead of 1/101~=0.0099). --n-half-splits left at
# 100 (unaffected -- that's the ceiling bootstrap, not the null test).
# One array task per (patient, model) pair; CPU-only, so high concurrency.

#SBATCH --job-name=othoth_rel_nnull1000
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=0-25%13
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/other_other_rel_nnull1000/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/other_other_rel_nnull1000/%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
mkdir -p logs/other_other_rel_nnull1000

PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"

PATIENTS=(
  PTYEU_task147
  PTYEY_task86
  PTYFA_task25
  PTYFC_task28
  PTYFF_task17
  PTYFG_task18
  PTYFK_task40
  PTYFM_task104
  PTYFP_task88
  PTYFR_task91
  PTYFS_task95
  PTYFU_task224
  PTYEZ_task60
)
N_PATIENTS=${#PATIENTS[@]}

model_idx=$(( SLURM_ARRAY_TASK_ID / N_PATIENTS ))
patient_idx=$(( SLURM_ARRAY_TASK_ID % N_PATIENTS ))
PATIENT="${PATIENTS[$patient_idx]}"

echo "task=${SLURM_ARRAY_TASK_ID} patient=${PATIENT} model_idx=${model_idx} host=$(hostname) started=$(date --iso-8601=seconds)"

if [[ "${model_idx}" == "0" ]]; then
    echo "--- no-tag ---"
    CONTEXT_TAG=""
    RESULTS_ROOT="/scratch/aniluchavez/ConvoDATAS/SemanticGLM/other_other_rcross_reliability_bert_pc50"
else
    echo "--- spktag ---"
    CONTEXT_TAG="-spktag"
    RESULTS_ROOT="/scratch/aniluchavez/ConvoDATAS/SemanticGLM/other_other_rcross_reliability_bert_spktag_pc50"
fi

"${PYTHON}" -u \
  scripts/other_other_rcross_reliability.py \
  --patient "${PATIENT}" \
  --region hippocampus \
  --model-tag bert-base-bidirmax \
  --context-tag="${CONTEXT_TAG}" \
  --window-tag tshift-150_tlen500_oshift+200_olen500 \
  --layer 12 \
  --n-components 50 \
  --n-null 1000 \
  --n-half-splits 100 \
  --n-jobs 8 \
  --results-root "${RESULTS_ROOT}"

echo "completed=$(date --iso-8601=seconds)"
