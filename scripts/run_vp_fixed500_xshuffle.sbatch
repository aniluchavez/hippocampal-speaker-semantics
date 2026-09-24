#!/bin/bash
#SBATCH --job-name=vp_xshuf
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_fixed500_xshuffle_sbatch.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_fixed500_xshuffle_sbatch.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/variance_partitioning.py
LOG_DIR=/scratch/aniluchavez/hippocampal-speaker-semantics/logs
mkdir -p "$LOG_DIR"

PATIENTS=(
    PTYEU_task147
    PTYFF_task17
    PTYFG_task18
    PTYFI_task81
    PTYFA_task25
    PTYFK_task40
    PTYEZ_task60
    PTYFC_task28
    PTYFM_task104
    PTYFP_task88
    PTYFR_task91
    PTYFS_task95
    PTYFU_task224
)

COMMON="--model gpt2-xl --context_tag _ctx200 --window_type tshift-200_tlen500_oshift+100_olen500 \
--layer 36 --n_components 100 --n_perm 50 \
--perm_type xshuffle --cv_mode shuffle --sym_perm"
BATCH_SIZE=5

echo "====== VP fixed-500ms, xshuffle+cvshuffle+sym_perm (13 non-sparse patients) ======"
for (( i=0; i<${#PATIENTS[@]}; i+=BATCH_SIZE )); do
    batch=("${PATIENTS[@]:$i:$BATCH_SIZE}")
    echo "--- batch $((i/BATCH_SIZE + 1)): ${batch[*]}"
    PIDS=()
    for PAT in "${batch[@]}"; do
        $PY -u $SCRIPT $COMMON --patient "$PAT" \
            > "$LOG_DIR/${PAT}_vp_fixed500_xshuffle.log" 2>&1 &
        PIDS+=($!)
        echo "  $PAT  PID=$!"
    done
    wait "${PIDS[@]}"
    echo "  batch done"
done

echo "ALL DONE"
