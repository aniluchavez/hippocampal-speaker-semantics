#!/usr/bin/env bash
# GPT-2 XL final layer (L48), low-PC sweep across matched CV strategies.
# Array = 3 PC counts x 4 CV strategies x 15 patients = 180 tasks.

#SBATCH --job-name=gpt2xl48_lowpc_cv
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-179%10
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48_lowpc_cv/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_gpt2xl_L48_lowpc_cv/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"

patients=(
    PTYEU_task147 PTYEV_task37 PTYEY_task86 PTYEZ_task60 PTYFA_task25
    PTYFC_task28 PTYFF_task17 PTYFG_task18 PTYFI_task81 PTYFK_task40
    PTYFM_task104 PTYFP_task88 PTYFR_task91 PTYFS_task95 PTYFU_task224
)
pcs=(5 10 20)
cv_modes=(shuffle purged300 purged500 block)

task="${SLURM_ARRAY_TASK_ID}"
patient_i=$((task % 15))
combo=$((task / 15))
cv_i=$((combo % 4))
pc_i=$((combo / 4))

patient="${patients[$patient_i]}"
pc="${pcs[$pc_i]}"
cv_label="${cv_modes[$cv_i]}"

case "${cv_label}" in
    shuffle)
        cv_args=(--outer_cv shuffle)
        ;;
    purged300)
        cv_args=(--outer_cv purged_shuffle --embargo_ms 300)
        ;;
    purged500)
        cv_args=(--outer_cv purged_shuffle --embargo_ms 500)
        ;;
    block)
        cv_args=(--outer_cv block)
        ;;
esac

cd "${PROJECT}"
echo "patient=${patient} pc=${pc} cv=${cv_label} host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --patient "${patient}" \
    --model gpt2-xl \
    --context_tag _ctx200 \
    --layer 48 \
    --n_components "${pc}" \
    --region hippocampus \
    --window_type fixed \
    --spike_tag bestfixed_selfm300_len500_otherp20_len500 \
    "${cv_args[@]}" \
    --n_outer 5 \
    --force_all \
    --notebook_exact \
    --r2_only

echo "completed=$(date --iso-8601=seconds)"
