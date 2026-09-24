#!/bin/bash
#SBATCH --job-name=null_cal_sym
#SBATCH --partition=guppy
#SBATCH --nodelist=guppy-1
#SBATCH --gres=gpu:l40:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/null_calibration_symmetric_sbatch.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/null_calibration_symmetric_sbatch.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/null_calibration_check_symmetric.py

$PY -u $SCRIPT
