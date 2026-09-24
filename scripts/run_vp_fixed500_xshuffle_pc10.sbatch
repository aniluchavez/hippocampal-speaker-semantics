#!/bin/bash
#SBATCH --job-name=vp_xshuf_pc10
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_fixed500_xshuffle_pc10_sbatch.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_fixed500_xshuffle_pc10_sbatch.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/variance_partitioning.py
LOG_DIR=/scratch/aniluchavez/hippocampal-speaker-semantics/logs
mkdir -p "$LOG_DIR"

COMMON="--model gpt2-xl --context_tag _ctx200 --window_type tshift-200_tlen500_oshift+100_olen500 \
--layer 36 --n_components 10 --n_perm 50 \
--perm_type xshuffle --cv_mode shuffle --sym_perm"

echo "====== VP fixed-500ms, xshuffle+cvshuffle+sym_perm — sparse patients pc10 ======"

$PY -u $SCRIPT $COMMON --patient PTYEV_task37 --min_spikes 1 \
    > "$LOG_DIR/PTYEV_task37_vp_fixed500_xshuffle_pc10.log" 2>&1 &
PID1=$!
echo "  PTYEV_task37 (min_spikes=1)  PID=$PID1"

$PY -u $SCRIPT $COMMON --patient PTYEY_task86 \
    > "$LOG_DIR/PTYEY_task86_vp_fixed500_xshuffle_pc10.log" 2>&1 &
PID2=$!
echo "  PTYEY_task86  PID=$PID2"

wait $PID1 $PID2
echo "ALL DONE"
