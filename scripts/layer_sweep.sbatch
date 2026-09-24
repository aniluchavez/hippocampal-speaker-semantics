#!/bin/bash
# Layer-wise semantic GLM sweep for llama-3.1-8b (every 4th layer, 0-32) and
# gpt2-xl (every 4th layer, 0-48), hippocampus only, worddur window, 50 perms,
# no --reliability (too expensive to run at every layer).
#
# Submit with:  sbatch scripts/layer_sweep.sbatch
# Adjust --partition/--account below for your cluster — these vary by site
# and aren't guessable from here.

#SBATCH --job-name=layer_sweep
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/layer_sweep_%j.out
#SBATCH --nodelist=guppy-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
##SBATCH --partition=<fill in if required>
##SBATCH --account=<fill in if required>

set -euo pipefail
cd /scratch/aniluchavez/hippocampal-speaker-semantics

source ~/miniforge3/etc/profile.d/conda.sh
conda activate gpt2_embed

mkdir -p logs/layer_sweep

# How many semantic_glm.py processes share the one GPU concurrently. Each fit
# uses ~20-40MB of GPU memory (out of ~49GB on the L40) and the CV/permutation
# loop is launch-overhead-bound rather than compute-bound at this problem size
# (n_components=20, tens of neurons, hundreds-to-thousands of words), so the
# GPU sits mostly idle between kernel launches in a single-process run.
# Running several processes at once fills that idle time instead of leaving
# it on the table. Start at 6; watch `nvidia-smi`/`htop` mid-run — push higher
# if GPU-Util stays low and CPU/RAM have headroom, pull back if jobs visibly
# slow each other down.
CONCURRENCY=6

JOBS_FILE=$(mktemp)
for L in 0 4 8 12 16 20 24 28 32; do
  echo "llama-3.1-8b _ctx200 $L" >> "$JOBS_FILE"
done
for L in 0 4 8 12 16 20 24 28 32 36 40 44 48; do
  echo "gpt2-xl _ctx200 $L" >> "$JOBS_FILE"
done

cat "$JOBS_FILE" | xargs -P "$CONCURRENCY" -L 1 bash -c '
  MODEL=$1; CTX=$2; LAYER=$3
  echo "=== $MODEL layer $LAYER starting $(date) ==="
  python3 -u scripts/semantic_glm.py \
    --model "$MODEL" --context_tag "$CTX" --layer "$LAYER" --n_components 20 \
    --region hippocampus --window_type worddur --outer_cv block --n_perm 50 \
    > "logs/layer_sweep/${MODEL}_L${LAYER}.log" 2>&1
  echo "=== $MODEL layer $LAYER done $(date) ==="
' bash

rm -f "$JOBS_FILE"
echo "All layer-sweep jobs complete."
