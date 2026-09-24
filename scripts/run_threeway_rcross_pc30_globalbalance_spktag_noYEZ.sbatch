#!/bin/bash
#SBATCH --job-name=rcross_pc30_spk_noYEZ
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/threeway_rcross_pc30_globalbalance_spktag_noYEZ.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/threeway_rcross_pc30_globalbalance_spktag_noYEZ.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/three_way_r_cross_pc30_globalbalance_spktag.py
LOG_DIR=/scratch/aniluchavez/hippocampal-speaker-semantics/logs
mkdir -p "$LOG_DIR"

$PY -u $SCRIPT ptYFK_task40 ptYFG_task18 ptYFF_task17 ptYFC_task28 ptYFA_task25 ptYEY_task86 ptYEU_task147 ptYFM_task104 ptYFP_task88 ptYFR_task91 ptYFS_task95 ptYFU_task224
