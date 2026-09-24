#!/bin/bash
# Rerun batch 1 selfother patients that were killed mid-perm

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/semantic_glm.py
LOG_DIR=/scratch/aniluchavez/hippocampal-speaker-semantics/logs

PATIENTS=(PTYEU_task147 PTYFF_task17 PTYFG_task18 PTYFI_task81 PTYFA_task25)
COMMON="--model gpt2-xl --context_tag _ctx200 --window_type worddur --layer 36 --n_perm 100 --outer_cv block"

echo "====== Rerunning batch 1 selfother ======"
PIDS=()
for PAT in "${PATIENTS[@]}"; do
    $PY -u $SCRIPT $COMMON --patient "$PAT" \
        > "$LOG_DIR/${PAT}_selfother_block.log" 2>&1 &
    PIDS+=($!)
    echo "  $PAT  PID=$!"
done
wait "${PIDS[@]}"
echo "ALL DONE"
