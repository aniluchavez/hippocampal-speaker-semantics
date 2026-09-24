#!/bin/bash
# PC sweep for robustness notebook (01_pc_robustness.ipynb)
# Runs real CV only (n_perm=0) across PC counts — fast, no perm refits needed.
# Output: gpt2-xl_ctx200_worddur_xcirc/pc{N}/ for each N in PC_COUNTS
# pc100 will be SKIPPED if xcirc run is already done (pkls exist).

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/semantic_glm.py
LOG_DIR=/scratch/aniluchavez/hippocampal-speaker-semantics/logs
mkdir -p "$LOG_DIR"

PATIENTS=(
    PTYEU_task147 PTYFF_task17 PTYFG_task18 PTYFI_task81 PTYFA_task25
    PTYFK_task40  PTYEY_task86 PTYEV_task37 PTYEZ_task60 PTYFC_task28
    PTYFM_task104 PTYFP_task88 PTYFR_task91 PTYFS_task95 PTYFU_task224
)

PC_COUNTS=(5 10 20 50 100 200)

BASE="--model gpt2-xl --context_tag _ctx200 --window_type worddur --layer 36 \
      --outer_cv block --perm_type xshuffle --n_perm 50 \
      --region hippocampus --force_all"
BATCH_SIZE=5

run_batch() {
    local pc=$1; shift
    local batch=("$@")
    local PIDS=()
    for PAT in "${batch[@]}"; do
        $PY -u $SCRIPT $BASE --n_components "$pc" --patient "$PAT" \
            > "$LOG_DIR/${PAT}_pc${pc}_sweep.log" 2>&1 &
        PIDS+=($!)
        echo "  pc=$pc  $PAT  PID=$!"
    done
    wait "${PIDS[@]}"
    echo "  batch done"
}

for PC in "${PC_COUNTS[@]}"; do
    echo "====== n_components=$PC ======"
    for (( i=0; i<${#PATIENTS[@]}; i+=BATCH_SIZE )); do
        batch=("${PATIENTS[@]:$i:$BATCH_SIZE}")
        echo "--- batch $((i/BATCH_SIZE + 1)): ${batch[*]}"
        run_batch "$PC" "${batch[@]}"
    done
done

echo "ALL DONE — run 01_pc_robustness.ipynb to plot results"
