#!/usr/bin/env bash

set -u

PROJECT_ROOT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
SCRIPT="$PROJECT_ROOT/scripts/cluster_glm_worddur_offset.py"
LOG_DIR="$PROJECT_ROOT/logs/cluster_glm_worddur_offset_remaining"
GPU_BLOCKING_PID="${GPU_BLOCKING_PID:-347613}"

PATIENTS=(
    PTYEU_task147
    PTYFF_task17
    PTYFG_task18
    PTYFI_task81
    PTYFA_task25
    PTYFK_task40
    PTYEV_task37
    PTYEZ_task60
    PTYFC_task28
    PTYFM_task104
    PTYFP_task88
    PTYFR_task91
    PTYFS_task95
    PTYFU_task224
)

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT" || exit 1

if kill -0 "$GPU_BLOCKING_PID" 2>/dev/null; then
    echo "Waiting for GPU-blocking PID $GPU_BLOCKING_PID to finish..."
    while kill -0 "$GPU_BLOCKING_PID" 2>/dev/null; do
        sleep 30
    done
fi

echo "Starting remaining-patient word-duration-offset batch."

failed=()
for patient in "${PATIENTS[@]}"; do
    log="$LOG_DIR/${patient}.log"
    echo "Starting $patient"
    if "$PYTHON" -u "$SCRIPT" --patient "$patient" >"$log" 2>&1; then
        echo "Completed $patient"
    else
        status=$?
        echo "Failed $patient with exit status $status; see $log"
        failed+=("$patient")
    fi
done

if ((${#failed[@]})); then
    echo "Batch completed with failures: ${failed[*]}"
    exit 1
fi

echo "Batch completed successfully for all 14 remaining patients."
