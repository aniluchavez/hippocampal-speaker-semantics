#!/bin/bash
#SBATCH --job-name=vp_bert_rem
#SBATCH --partition=guppy
#SBATCH --nodelist=guppy-1
#SBATCH --gres=gpu:l40:1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_bert_xcirc_pc30_symperm_remaining.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/vp_bert_xcirc_pc30_symperm_remaining.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
SCRIPT=/scratch/aniluchavez/hippocampal-speaker-semantics/scripts/variance_partitioning.py
LOG_DIR=/scratch/aniluchavez/hippocampal-speaker-semantics/logs

COMMON="--model bert-base --context_tag _ctx200 --window_type worddur \
        --layer 12 --n_perm 50 --n_components 30 \
        --perm_type xcirc --cv_mode block --sym_perm"

PATIENTS=(
    PTYEU_task147
    PTYFA_task25
    PTYFF_task17
    PTYFG_task18
    PTYFI_task81
    PTYFM_task104
    PTYFP_task88
    PTYFR_task91
    PTYFS_task95
    PTYFU_task224
)
BATCH_SIZE=5

echo "====== VP bert-base L12 xcirc pc30 remaining ======"
for (( i=0; i<${#PATIENTS[@]}; i+=BATCH_SIZE )); do
    batch=("${PATIENTS[@]:$i:$BATCH_SIZE}")
    echo "--- batch $((i/BATCH_SIZE + 1)): ${batch[*]}"
    PIDS=()
    for PAT in "${batch[@]}"; do
        $PY -u $SCRIPT $COMMON --patient "$PAT" \
            > "$LOG_DIR/${PAT}_vp_bert_xcirc_pc30_symperm.log" 2>&1 &
        PIDS+=($!)
        echo "  $PAT  PID=$!"
    done
    wait "${PIDS[@]}"
    echo "  batch done"
done
echo "ALL DONE"
