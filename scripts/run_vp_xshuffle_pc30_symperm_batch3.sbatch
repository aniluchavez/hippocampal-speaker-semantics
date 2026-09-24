#!/bin/bash
#SBATCH --job-name=vp_xshuf_b3
#SBATCH --partition=guppy
#SBATCH --nodelist=guppy-4
#SBATCH --gres=gpu:l40:1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_xshuffle_pc30_symperm_batch3.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_xshuffle_pc30_symperm_batch3.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/variance_partitioning.py
LOG_DIR=/scratch/aniluchavez/hippocampal-speaker-semantics/logs

COMMON="--model gpt2-xl --context_tag _ctx200 --window_type worddur \
        --layer 36 --n_perm 50 --n_components 30 \
        --perm_type xshuffle --cv_mode shuffle --sym_perm"

PATIENTS=(PTYFM_task104 PTYFP_task88 PTYFR_task91 PTYFS_task95 PTYFU_task224)

echo "====== VP xshuffle pc30 batch3 ======"
PIDS=()
for PAT in "${PATIENTS[@]}"; do
    $PY -u $SCRIPT $COMMON --patient "$PAT" \
        > "$LOG_DIR/${PAT}_vp_xshuffle_pc30_symperm.log" 2>&1 &
    PIDS+=($!)
    echo "  $PAT  PID=$!"
done
wait "${PIDS[@]}"
echo "ALL DONE"
