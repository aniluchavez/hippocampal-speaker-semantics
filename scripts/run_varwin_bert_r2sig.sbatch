#!/usr/bin/env bash
# Point 2 gap-filler: BERT (causal, spktag) overall encoding strength (R2 +
# % significant) under variable (varwin) windows, matching the exact window
# used by run_varwin_reliability_models.sbatch's r_cross analysis (self:
# onset-300ms to offset; other: onset to offset+100ms), layer 12, pc=50.
#
# Unlike that reliability job, this one is NOT --r2_only (need real xcirc
# permutation nulls for %significant) and skips --reliability (don't need
# r_cross here, just the plain encoding-strength test used throughout).
# n_perm=100, matching today's other null-check jobs.

#SBATCH --job-name=varwin_bert_r2sig
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/varwin_bert_r2sig/%j.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/varwin_bert_r2sig/%j.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
SPIKE_TAG="varwin_selfm300tooffset0_other0tooffsetp100"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${PROJECT}"
mkdir -p logs/varwin_bert_r2sig

echo "model=bert-base-causal_ctx200spktag layer=12 pc=50 (varwin, n_perm=100)"
echo "window self=[onset-300, offset], other=[onset, offset+100]"
echo "host=$(hostname) started=$(date --iso-8601=seconds)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model bert-base-causal \
    --context_tag _ctx200spktag \
    --layer 12 \
    --n_components 50 \
    --region hippocampus \
    --window_type varwin \
    --spike_tag "${SPIKE_TAG}" \
    --outer_cv shuffle \
    --force_all \
    --notebook_exact \
    --perm_type xcirc \
    --n_perm 100

echo "completed=$(date --iso-8601=seconds)"
