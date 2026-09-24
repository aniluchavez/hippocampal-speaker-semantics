#!/bin/bash
#SBATCH --job-name=resid_op_bert
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/resid_rcross_op_bert.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/resid_rcross_op_bert.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/compute_residualized_rcross_other_pooled.py

$PY -u $SCRIPT --model bert-base --context_tag _ctx200 --layer 12 --n_components 30
