#!/usr/bin/env bash
# GPT-2 Large L36, 30 PCs, best fixed windows, distributed random K-fold CV
# with a configurable temporal embargo around every outer and inner test word.
# Submit with EMBARGO_MS=500 for a strict no-overlap check with 500 ms windows.

#SBATCH --job-name=gpt2L_pc30_purge
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-14%5
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2large_L36_pc30_purged/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2large_L36_pc30_purged/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
EMBARGO_MS="${EMBARGO_MS:-300}"

patients=(
    PTYEU_task147 PTYEV_task37 PTYEY_task86 PTYEZ_task60 PTYFA_task25
    PTYFC_task28 PTYFF_task17 PTYFG_task18 PTYFI_task81 PTYFK_task40
    PTYFM_task104 PTYFP_task88 PTYFR_task91 PTYFS_task95 PTYFU_task224
)
patient="${patients[$SLURM_ARRAY_TASK_ID]}"

cd "${PROJECT}"
mkdir -p logs/fixed_gpt2large_L36_pc30_purged

echo "patient=${patient} host=$(hostname) started=$(date --iso-8601=seconds)"
echo "CV=purged_shuffle outer=5 inner=5 embargo=${EMBARGO_MS}ms"
echo "windows=self[-300,+200]ms other[+20,+520]ms"

"${PYTHON}" -u scripts/semantic_glm.py \
    --patient "${patient}" \
    --model gpt2-large \
    --context_tag _ctx200 \
    --layer 36 \
    --n_components 30 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag bestfixed_selfm300_len500_otherp20_len500 \
    --outer_cv purged_shuffle \
    --embargo_ms "${EMBARGO_MS}" \
    --n_outer 5 \
    --force_all \
    --notebook_exact \
    --r2_only

echo "completed=$(date --iso-8601=seconds)"
