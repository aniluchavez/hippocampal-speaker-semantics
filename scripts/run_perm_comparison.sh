#!/bin/bash
# Run xcirc and xblock permutations for hippocampus self+other, all 15 patients
# Output dirs:
#   xcirc  → gpt2-xl_ctx200_worddur_xcirc/pc100
#   xblock → gpt2-xl_ctx200_worddur_xblock/pc100

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/semantic_glm.py
LOG_DIR=/scratch/aniluchavez/hippocampal-speaker-semantics/logs
mkdir -p "$LOG_DIR"

PATIENTS=(
    PTYEU_task147 PTYFF_task17 PTYFG_task18 PTYFI_task81 PTYFA_task25
    PTYFK_task40  PTYEY_task86 PTYEV_task37 PTYEZ_task60 PTYFC_task28
    PTYFM_task104 PTYFP_task88 PTYFR_task91 PTYFS_task95 PTYFU_task224
)

BASE="--model gpt2-xl --context_tag _ctx200 --window_type worddur --layer 36 --n_perm 500 --outer_cv block --region hippocampus --force_all"
BATCH_SIZE=5

run_batch() {
    local perm_type=$1; shift
    local batch=("$@")
    local PIDS=()
    for PAT in "${batch[@]}"; do
        $PY -u $SCRIPT $BASE --perm_type "$perm_type" --patient "$PAT" \
            > "$LOG_DIR/${PAT}_hippo_selfother_${perm_type}.log" 2>&1 &
        PIDS+=($!)
        echo "  [$perm_type] $PAT  PID=$!"
    done
    wait "${PIDS[@]}"
    echo "  batch done"
}

for PERM in xcirc xblock; do
    echo "====== PERM_TYPE: $PERM ======"
    for (( i=0; i<${#PATIENTS[@]}; i+=BATCH_SIZE )); do
        batch=("${PATIENTS[@]:$i:$BATCH_SIZE}")
        echo "--- batch $((i/BATCH_SIZE + 1)): ${batch[*]}"
        run_batch "$PERM" "${batch[@]}"
    done
done

echo "ALL DONE"
