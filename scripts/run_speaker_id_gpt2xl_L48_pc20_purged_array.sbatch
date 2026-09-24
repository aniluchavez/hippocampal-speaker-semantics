#!/usr/bin/env bash
# Unique speaker-identity encoding beyond GPT-2 XL semantics.
# Common 0--500 ms fixed window for every speaker.
# Array = 15 patients x embargoes {300, 500} ms.

#SBATCH --job-name=spkid_xl48_pc20
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-29%6
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/speaker_id_gpt2xl_L48_pc20_purged/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/speaker_id_gpt2xl_L48_pc20_purged/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"

patients=(
    PTYEU_task147 PTYEV_task37 PTYEY_task86 PTYEZ_task60 PTYFA_task25
    PTYFC_task28 PTYFF_task17 PTYFG_task18 PTYFI_task81 PTYFK_task40
    PTYFM_task104 PTYFP_task88 PTYFR_task91 PTYFS_task95 PTYFU_task224
)
embargoes=(300 500)

task="${SLURM_ARRAY_TASK_ID}"
patient="${patients[$((task % 15))]}"
embargo="${embargoes[$((task / 15))]}"

cd "${PROJECT}"
echo "patient=${patient} embargo=${embargo}ms host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/joint_speaker_semantic_glm.py \
    --patient "${patient}" \
    --model gpt2-xl \
    --context_tag _ctx200 \
    --layer 48 \
    --n_components 20 \
    --n_perm 100 \
    --spike_tag fixed_self0_other0_len500 \
    --window_type fixed \
    --cv_mode purged_shuffle \
    --embargo_ms "${embargo}" \
    --min_speaker_words 100 \
    --min_spikes 20

echo "completed=$(date --iso-8601=seconds)"
