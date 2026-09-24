#!/usr/bin/env bash
# Re-run PTYEY_task86 only, with --force_all, for the worddur/xcirc BERT L12
# pc20 table -- same fix as the xshuffle counterpart (job 560896): only 67
# words survive word-duration filtering for this patient's self condition,
# which trips the "too few samples" skip unless --force_all is passed.

#SBATCH --job-name=worddur_bert_L12_pc20_xcirc_ptyey
#SBATCH --partition=mammoth
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/worddur_bert_L12_pc20/xcirc_ptyey_forceall_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/worddur_bert_L12_pc20/xcirc_ptyey_forceall_%j.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${PROJECT}"
mkdir -p logs/worddur_bert_L12_pc20

echo "model=bert-base layer=12 pc=20 (worddur window, SHUFFLE cv, xcirc null, n_perm=100), patient=PTYEY_task86, --force_all"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model bert-base \
    --context_tag _ctx200 \
    --layer 12 \
    --n_components 20 \
    --region hippocampus \
    --window_type worddur \
    --outer_cv shuffle \
    --perm_type xcirc \
    --n_perm 100 \
    --patient PTYEY_task86 \
    --force_all

echo "completed=$(date --iso-8601=seconds)"
