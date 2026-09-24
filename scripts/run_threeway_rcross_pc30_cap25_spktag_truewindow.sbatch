#!/bin/bash
#SBATCH --job-name=rcross_pc30_cap25_spk_tw
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/threeway_rcross_pc30_cap25_spktag_truewindow.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/threeway_rcross_pc30_cap25_spktag_truewindow.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/three_way_r_cross_pc30_cap25_spktag_truewindow.py
LOG_DIR=/scratch/aniluchavez/hippocampal-speaker-semantics/logs
mkdir -p "$LOG_DIR"

$PY -u $SCRIPT
