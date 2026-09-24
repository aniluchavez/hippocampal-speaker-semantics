#!/bin/bash
#SBATCH --job-name=vp_fixed500_tm200_o100
#SBATCH --partition=guppy
#SBATCH --nodelist=guppy-1
#SBATCH --gres=gpu:l40:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_fixed500_tm200_o100_sbatch.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_fixed500_tm200_o100_sbatch.err

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
    PTYEY_task86
    PTYEV_task37
    PTYEZ_task60
    PTYFC_task28
    PTYFM_task104
    PTYFP_task88
    PTYFR_task91
    PTYFS_task95
    PTYFU_task224
)

# fixed 500ms windows: self [-200ms, +300ms], other [+100ms, +600ms] from word onset
COMMON="--model gpt2-xl --context_tag _ctx200 --window_type tshift-200_tlen500_oshift+100_olen500 --layer 36 --n_components 100 --n_perm 50 --perm_type xcirc --cv_mode block"
BATCH_SIZE=5

echo "====== VP fixed-500ms window (self -200ms, other +100ms), xcirc null + block CV (all patients) ======"
for (( i=0; i<${#PATIENTS[@]}; i+=BATCH_SIZE )); do
    batch=("${PATIENTS[@]:$i:$BATCH_SIZE}")
    echo "--- batch $((i/BATCH_SIZE + 1)): ${batch[*]}"
    PIDS=()
    for PAT in "${batch[@]}"; do
        $PY -u $SCRIPT $COMMON --patient "$PAT" \
            > "$LOG_DIR/${PAT}_vp_fixed500_tm200_o100.log" 2>&1 &
        PIDS+=($!)
        echo "  $PAT  PID=$!"
    done
    wait "${PIDS[@]}"
    echo "  batch done"
done

echo "ALL DONE"
