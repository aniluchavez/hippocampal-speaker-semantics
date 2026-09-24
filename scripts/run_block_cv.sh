#!/bin/bash
# Temporal block CV (5-fold) run — all 15 patients, self/other + all_conditions
set -e

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

total=${#PATIENTS[@]}
idx=0
for PAT in "${PATIENTS[@]}"; do
    idx=$((idx+1))
    echo "=========================================="
    echo "[$idx/$total] $PAT  — self/other"
    echo "=========================================="
    $PY -u $SCRIPT $COMMON --patient "$PAT" 2>&1 | tee "$LOG_DIR/${PAT}_selfother_block.log"

    echo "=========================================="
    echo "[$idx/$total] $PAT  — all_conditions"
    echo "=========================================="
    $PY -u $SCRIPT $COMMON --patient "$PAT" --all_conditions 2>&1 | tee "$LOG_DIR/${PAT}_allcond_block.log"
done

echo "ALL DONE"
