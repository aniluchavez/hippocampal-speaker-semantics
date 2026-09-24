#!/bin/bash
# Run ctx200 GLM layer sweeps for all models.
# Each model's layers run in parallel; models are sequential (prevents GPU OOM).
# Usage: nohup bash scripts/run_ctx200_sweeps.sh > /tmp/ctx200_sweeps.log 2>&1 &

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics

SCRIPT="scripts/semantic_glm.py"
BASE="--n_components 20 --n_perm 500 --context_tag _ctx200"

run_model() {
    local model=$1
    local n_layers=$2
    echo ""
    echo "========================================"
    echo "MODEL: $model  ($n_layers layers)  $(date)"
    echo "========================================"
    for L in $(seq 0 $((n_layers - 1))); do
        conda run -n gpt2_embed python3 -u $SCRIPT \
            --model $model --layer $L $BASE \
            > /tmp/sem_ctx200_${model}_L$(printf '%02d' $L).log 2>&1 &
    done
    wait   # wait for all layers of this model before starting next
    echo "  $model done  $(date)"
}

# Small/fast models first, gpt2-xl last (largest embeddings)
run_model gpt2             13
run_model bert-base        13
run_model bert-base-causal 13
run_model opt-350m         24
run_model gpt2-medium      25
run_model llama-2-7b       33
run_model llama-3.1-8b     33
run_model gpt2-xl          49

echo ""
echo "All ctx200 sweeps complete  $(date)"
