#!/usr/bin/env bash
# Same as run_fixed_bert_L12_pc20_shufflecv_xcirc.sbatch but --perm_type
# xshuffle instead of xcirc, for the "Sig.\ xshuffle" column of the
# fixed_vs_worddur_encoding_matched_table (BERT counterpart).

#SBATCH --job-name=fixed_bert_L12_pc20_xshuf
#SBATCH --partition=mammoth
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_bert_L12_pc20/xshuffle_%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/fixed_bert_L12_pc20/xshuffle_%j.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
SPIKE_TAG="fixed_selfm200_otherp100_len500"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${PROJECT}"
mkdir -p logs/fixed_bert_L12_pc20

echo "model=bert-base layer=12 pc=20 (fixed window, xshuffle null, n_perm=100)"
echo "window self=[onset-200, onset+300], other=[onset+100, onset+600]"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model bert-base \
    --context_tag _ctx200 \
    --layer 12 \
    --n_components 20 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag "${SPIKE_TAG}" \
    --outer_cv shuffle \
    --force_all \
    --notebook_exact \
    --perm_type xshuffle \
    --n_perm 100

echo "completed=$(date --iso-8601=seconds)"
