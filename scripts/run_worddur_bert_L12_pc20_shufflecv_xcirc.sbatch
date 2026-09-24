#!/usr/bin/env bash
# BERT L12 counterpart of run_worddur_gpt2xl_L48_shufflecv_xcirc.sbatch, for
# the word-duration side of the fixed_vs_worddur_encoding_matched_table.
# Duration is used as a Poisson exposure offset (--window_type worddur).
# No --spike_tag override, no --notebook_exact, no --force_all -- matches
# the GPT2-XL worddur invocation exactly (worddur spike windows already
# exist for all 15 patients under the default "worddur" tag).

#SBATCH --job-name=worddur_bert_L12_pc20_xcirc
#SBATCH --partition=mammoth
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/worddur_bert_L12_pc20/xcirc_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/worddur_bert_L12_pc20/xcirc_%j.err

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

echo "model=bert-base layer=12 pc=20 (worddur window, SHUFFLE cv, xcirc null, n_perm=100)"
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
    --n_perm 100

echo "completed=$(date --iso-8601=seconds)"
