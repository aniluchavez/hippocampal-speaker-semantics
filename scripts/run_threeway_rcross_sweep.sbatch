#!/bin/bash
#SBATCH --job-name=rcross_sweep
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --array=0-6
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/rcross_sweep_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/rcross_sweep_%a.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/three_way_r_cross_globalbalance_spktag_sweep.py
LOG_DIR=/scratch/aniluchavez/hippocampal-speaker-semantics/logs
mkdir -p "$LOG_DIR"

TRUE_WINDOW="tshift-150_tlen500_oshift+200_olen500"

case $SLURM_ARRAY_TASK_ID in
  # --- PC sweep at the true window (10/30/100 already done separately) ---
  0) ARGS="--spike_tag $TRUE_WINDOW --n_components 20 --out_tag pc20_trueWindow" ;;
  1) ARGS="--spike_tag $TRUE_WINDOW --n_components 50 --out_tag pc50_trueWindow" ;;
  2) ARGS="--spike_tag $TRUE_WINDOW --n_components 75 --out_tag pc75_trueWindow" ;;
  # --- window sweep at PC=30 (true window -150/+200 already done separately) ---
  3) ARGS="--spike_tag fixed_selfm100_otherp200_len500 --n_components 30 --out_tag pc30_selfm100_otherp200" ;;
  4) ARGS="--spike_tag fixed_selfm400_otherp200_len500 --n_components 30 --out_tag pc30_selfm400_otherp200" ;;
  5) ARGS="--spike_tag fixed_selfm300_otherp200_len500 --n_components 30 --out_tag pc30_selfm300_otherp200" ;;
  6) ARGS="--spike_tag tshift-200_tlen500_oshift+100_olen500 --n_components 30 --out_tag pc30_selfm200_otherp100" ;;
esac

echo "Task $SLURM_ARRAY_TASK_ID: $ARGS"
$PY -u $SCRIPT $ARGS
