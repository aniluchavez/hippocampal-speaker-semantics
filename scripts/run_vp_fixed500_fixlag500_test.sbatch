#!/bin/bash
#SBATCH --job-name=vp_fixlag
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_fixlag500_test.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_fixlag500_test.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/variance_partitioning.py
LOG_DIR=/scratch/aniluchavez/hippocampal-speaker-semantics/logs
mkdir -p "$LOG_DIR"

# Quick test: one patient, xcirc with fixed ±500ms lag (paper-style null), block CV, sym_perm
$PY -u $SCRIPT \
    --model gpt2-xl --context_tag _ctx200 \
    --window_type tshift-200_tlen500_oshift+100_olen500 \
    --layer 36 --n_components 100 --n_perm 50 \
    --perm_type xcirc --cv_mode block --sym_perm \
    --fixed_lag_ms 500 \
    --patient PTYEU_task147 \
    > "$LOG_DIR/PTYEU_task147_vp_fixlag500.log" 2>&1

echo "DONE"
