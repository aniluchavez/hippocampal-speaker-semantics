#!/usr/bin/env bash
#SBATCH --job-name=ml_xshuf
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/model_layer_sweep/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/model_layer_sweep/%A_%a.err

set -euo pipefail

PROJECT="/scratch/aniluchavez/hippocampal-speaker-semantics"
PYTHON="/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3"
CONFIG="/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/best_fixed_window_all15.json"

models=(gpt2-large gpt2-xl llama-3.1-8b bert-base fasttext-wiki)
contexts=(_ctx200 _ctx200 _ctx200 _ctx200 "")
layer_counts=(37 49 33 13 1)

task="${SLURM_ARRAY_TASK_ID}"
offset=0
model=""
context=""
layer=""
for i in "${!models[@]}"; do
    count="${layer_counts[$i]}"
    if (( task < offset + count )); then
        model="${models[$i]}"
        context="${contexts[$i]}"
        layer=$((task - offset))
        break
    fi
    offset=$((offset + count))
done

if [[ -z "${model}" ]]; then
    echo "Invalid task index: ${task}" >&2
    exit 2
fi

spike_tag="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["spike_tag"])' "${CONFIG}")"

mkdir -p /scratch/aniluchavez/hippocampal-speaker-semantics/logs/model_layer_sweep

cd "${PROJECT}"
echo "task=${task} model=${model} context=${context} layer=${layer} spike_tag=${spike_tag} host=$(hostname)"

"${PYTHON}" -u scripts/semantic_glm.py \
    --model "${model}" \
    --context_tag "${context}" \
    --layer "${layer}" \
    --n_components 50 \
    --region hippocampus \
    --window_type fixed \
    --spike_tag "${spike_tag}" \
    --outer_cv shuffle \
    --perm_type xshuffle \
    --n_perm 100 \
    --force_all \
    --notebook_exact

echo "completed=$(date --iso-8601=seconds)"
