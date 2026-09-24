#!/usr/bin/env bash
# Matched random or contiguous-block baseline for the 30-PC purged-CV test.
# Submit with: sbatch --export=ALL,CV_MODE=shuffle|block this_script.sbatch

#SBATCH --job-name=gpt2L_pc30_cv
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-14%5
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2large_L36_pc30_matchedcv/%x_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2large_L36_pc30_matchedcv/%x_%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
CV_MODE="${CV_MODE:?Submit with CV_MODE=shuffle or CV_MODE=block}"
if [[ "${CV_MODE}" != "shuffle" && "${CV_MODE}" != "block" ]]; then
    echo "CV_MODE must be shuffle or block, got ${CV_MODE}" >&2
    exit 2
fi

patients=(
    PTYEU_task147 PTYEV_task37 PTYEY_task86 PTYEZ_task60 PTYFA_task25
    PTYFC_task28 PTYFF_task17 PTYFG_task18 PTYFI_task81 PTYFK_task40
    PTYFM_task104 PTYFP_task88 PTYFR_task91 PTYFS_task95 PTYFU_task224
)
patient="${patients[$SLURM_ARRAY_TASK_ID]}"

cd "${PROJECT}"
echo "patient=${patient} CV=${CV_MODE} host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --patient "${patient}" \
    --model gpt2-large \
    --context_tag _ctx200 \
    --layer 36 \
    --n_components 30 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag bestfixed_selfm300_len500_otherp20_len500 \
    --outer_cv "${CV_MODE}" \
    --n_outer 5 \
    --force_all \
    --notebook_exact \
    --r2_only

echo "completed=$(date --iso-8601=seconds)"
