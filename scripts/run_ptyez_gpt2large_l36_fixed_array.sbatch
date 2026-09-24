#!/usr/bin/env bash
#SBATCH --job-name=ptyez_gpt2l
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/ptyez_gpt2large_%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/ptyez_gpt2large_%A_%a.err

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics
PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
TAG=tref-offset_shift-200_tlen500_oref-onset_shift+100_olen500
BASE=/scratch/aniluchavez/ConvoDATAS/SemanticGLM
COMMON=(
  --patient PTYEZ_task60
  --region hippocampus
  --n_jobs 8
  --window-tag "$TAG"
  --model-tag gpt2-large
  --layer 36
)

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  "$PY" -u scripts/cluster_glm_legacy_replication.py \
    "${COMMON[@]}" \
    --no-balance \
    --results-root "$BASE/clusterwise_gpt2large_L36_fixed_self_offsetm200_other_onsetp100_alldata"
else
  "$PY" -u scripts/cluster_glm_legacy_replication.py \
    "${COMMON[@]}" \
    --exclude-clusters 2 4 7 \
    --hard-balance \
    --results-root "$BASE/clusterwise_gpt2large_L36_fixed_self_offsetm200_other_onsetp100_hardbalanced47"
fi
