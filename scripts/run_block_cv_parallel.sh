#!/bin/bash
# Parallel block CV — patients in batches of BATCH_SIZE on shared GPU
# Each batch runs simultaneously; batches are sequential

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/semantic_glm.py
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

COMMON="--model gpt2-xl --context_tag _ctx200 --window_type worddur --layer 36 --n_perm 100 --outer_cv block"
BATCH_SIZE=5

run_batch() {
    local mode=$1; shift
    local batch=("$@")
    local PIDS=()
    for PAT in "${batch[@]}"; do
        if [ "$mode" = "allcond" ]; then
            $PY -u $SCRIPT $COMMON --patient "$PAT" --all_conditions \
                > "$LOG_DIR/${PAT}_allcond_block.log" 2>&1 &
        else
            $PY -u $SCRIPT $COMMON --patient "$PAT" \
                > "$LOG_DIR/${PAT}_selfother_block.log" 2>&1 &
        fi
        PIDS+=($!)
        echo "  [$mode] $PAT  PID=$!"
    done
    wait "${PIDS[@]}"
    echo "  batch done"
}

for mode in selfother allcond; do
    echo "====== MODE: $mode ======"
    for (( i=0; i<${#PATIENTS[@]}; i+=BATCH_SIZE )); do
        batch=("${PATIENTS[@]:$i:$BATCH_SIZE}")
        echo "--- batch $((i/BATCH_SIZE + 1)): ${batch[*]}"
        run_batch "$mode" "${batch[@]}"
    done
done

echo "ALL DONE"
