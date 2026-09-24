#!/usr/bin/env bash
# Table S6 reverification: BERT L12, bestfixed window (selfm300_len500_otherp20_len500),
# pc50, 15 patients x 3 CV/null combinations, with permutation testing actually enabled
# (no --r2_only) so p_perm is real instead of the placeholder 1.0 found in the existing
# cache. Combos:
#   combo 0: Block CV,   circular null  (--outer_cv block   --perm_type xcirc)
#   combo 1: Block CV,   shuffled null  (--outer_cv block   --perm_type xshuffle)
#   combo 2: Shuffle CV, shuffled null  (--outer_cv shuffle --perm_type xshuffle)
# One array task per (patient, combo) pair. Output directories are auto-named by
# semantic_glm.py from outer_cv/perm_type and will NOT collide with the existing
# r2only cache.

#SBATCH --job-name=table_s6_sweep
#SBATCH --partition=guppy,mammoth
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=0-44%8
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/table_s6_sweep/%x_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/table_s6_sweep/%x_%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
mkdir -p logs/table_s6_sweep

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

combo_idx=$(( SLURM_ARRAY_TASK_ID / N_PATIENTS ))
patient_idx=$(( SLURM_ARRAY_TASK_ID % N_PATIENTS ))
PATIENT="${PATIENTS[$patient_idx]}"

case "${combo_idx}" in
  0) OUTER_CV="block";   PERM_TYPE="xcirc";    LABEL="Block CV, circular null" ;;
  1) OUTER_CV="block";   PERM_TYPE="xshuffle"; LABEL="Block CV, shuffled null" ;;
  2) OUTER_CV="shuffle"; PERM_TYPE="xshuffle"; LABEL="Shuffle CV, shuffled null" ;;
  *) echo "bad combo_idx ${combo_idx}"; exit 1 ;;
esac

echo "task=${SLURM_ARRAY_TASK_ID} patient=${PATIENT} combo=${LABEL} host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model bert-base \
    --context_tag _ctx200 \
    --layer 12 \
    --n_components 50 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag "${SPIKE_TAG}" \
    --outer_cv "${OUTER_CV}" \
    --perm_type "${PERM_TYPE}" \
    --force_all \
    --notebook_exact \
    --patient "${PATIENT}"

echo "completed=$(date --iso-8601=seconds)"
