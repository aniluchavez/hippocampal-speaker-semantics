#!/bin/bash
#SBATCH --job-name=vp_fixed500_symperm_pc10
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_fixed500_symperm_pc10_sbatch.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_fixed500_symperm_pc10_sbatch.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/variance_partitioning.py
LOG_DIR=/scratch/aniluchavez/hippocampal-speaker-semantics/logs
mkdir -p "$LOG_DIR"

COMMON="--model gpt2-xl --context_tag _ctx200 --window_type tshift-200_tlen500_oshift+100_olen500 --layer 36 --n_components 10 --n_perm 50 --perm_type xcirc --cv_mode block --sym_perm"

for PAT in PTYEV_task37 PTYEY_task86; do
    $PY -u $SCRIPT $COMMON --patient "$PAT" \
        > "$LOG_DIR/${PAT}_vp_fixed500_symperm_pc10.log" 2>&1 &
done
wait
echo "ALL DONE"
