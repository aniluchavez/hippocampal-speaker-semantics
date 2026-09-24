#!/bin/bash
#SBATCH --job-name=bert_ll_diff
#SBATCH --output=logs/bert_ll_diff_%j.out
#SBATCH --error=logs/bert_ll_diff_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd /scratch/aniluchavez/hippocampal-speaker-semantics
mkdir -p logs

/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 \
  scripts/summarize_bestfixed_bert_ll_diff.py
