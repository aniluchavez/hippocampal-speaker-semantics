#!/bin/bash
#SBATCH --job-name=vp_fixed500_symperm
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_fixed500_symperm_sbatch.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_fixed500_symperm_sbatch.err

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

# Fixed 500ms window, xcirc null + block CV + symmetric ctrl permutation
# Output dir: gpt2-xl_ctx200_tshift-200_tlen500_oshift+100_olen500_xcirc_symperm/pc100
COMMON="--model gpt2-xl --context_tag _ctx200 --window_type tshift-200_tlen500_oshift+100_olen500 --layer 36 --n_components 100 --n_perm 50 --perm_type xcirc --cv_mode block --sym_perm"
BATCH_SIZE=5

echo "====== VP fixed-500ms, xcirc+block+sym_perm (all 15 patients) ======"
for (( i=0; i<${#PATIENTS[@]}; i+=BATCH_SIZE )); do
    batch=("${PATIENTS[@]:$i:$BATCH_SIZE}")
    echo "--- batch $((i/BATCH_SIZE + 1)): ${batch[*]}"
    PIDS=()
    for PAT in "${batch[@]}"; do
        $PY -u $SCRIPT $COMMON --patient "$PAT" \
            > "$LOG_DIR/${PAT}_vp_fixed500_symperm.log" 2>&1 &
        PIDS+=($!)
        echo "  $PAT  PID=$!"
    done
    wait "${PIDS[@]}"
    echo "  batch done"
done

echo "ALL DONE"
