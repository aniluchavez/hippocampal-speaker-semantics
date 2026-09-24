#!/bin/bash
#SBATCH --job-name=rcross_pc30_global
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/threeway_rcross_pc30_globalbalance.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/threeway_rcross_pc30_globalbalance.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/three_way_r_cross_pc30_globalbalance.py
LOG_DIR=/scratch/aniluchavez/hippocampal-speaker-semantics/logs
mkdir -p "$LOG_DIR"

$PY -u $SCRIPT ptYFK_task40 ptYFG_task18 ptYFF_task17 ptYFM_task104 ptYFP_task88 ptYFU_task224 ptYEZ_task60
