#!/bin/bash
#SBATCH --job-name=resid_oo_gpt2xl
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/resid_rcross_oo_gpt2xl.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/resid_rcross_oo_gpt2xl.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/compute_residualized_rcross_other_other.py

$PY -u $SCRIPT --model gpt2-xl --context_tag _ctx200 --layer 36 --n_components 30
