#!/usr/bin/env bash
set -euo pipefail

# Nohup launcher for fixed- or variable-window broad sweeps.
#
# Defaults run the same-length fixed-window broad search:
#   lengths:      200, 300, 500 ms
#   self starts:  -500, -400, -300, -200, -100, 0 ms
#   other starts: 0, +20, +100, +200 ms
#
# Override any parameter by setting an environment variable before bash:
#   LENGTHS=200,300 SELF_STARTS=-400,-300,-200 OTHER_STARTS=20,100 \
#     bash scripts/run_fixed_window_broad_sweep_nohup.sh
#
# Variable windows run from onset + START to offset + END_SHIFT:
#   WINDOW_MODE=varwin SELF_STARTS=-400,-200,0 OTHER_STARTS=0,20,100 \
#     SELF_END_SHIFTS=0 OTHER_END_SHIFTS=0,100 \
#     bash scripts/run_fixed_window_broad_sweep_nohup.sh

PROJECT="${PROJECT:-/scratch/aniluchavez/hippocampal-speaker-semantics}"
PYTHON="${PYTHON:-/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3}"

PATIENT="${PATIENT:-PTYEU_task147}"
REGION="${REGION:-hippocampus}"
MODEL="${MODEL:-gpt2-large}"
CONTEXT_TAG="${CONTEXT_TAG:-_ctx200}"
LAYER="${LAYER:-36}"
PC="${PC:-50}"

WINDOW_MODE="${WINDOW_MODE:-fixed}"
LENGTHS="${LENGTHS:-200,300,500}"
SELF_STARTS="${SELF_STARTS:--500,-400,-300,-200,-100,0}"
OTHER_STARTS="${OTHER_STARTS:-0,20,100,200}"
SELF_END_SHIFTS="${SELF_END_SHIFTS:-0}"
OTHER_END_SHIFTS="${OTHER_END_SHIFTS:-100}"

LOG_DIR="${LOG_DIR:-/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots}"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/${PATIENT}_${MODEL//-/}_L${LAYER}_pc${PC}_${WINDOW_MODE}_broad_sweep_${STAMP}.log"

cd "${PROJECT}"

echo "Launching ${WINDOW_MODE}-window broad sweep"
echo "  patient=${PATIENT}"
echo "  region=${REGION}"
echo "  model=${MODEL}${CONTEXT_TAG}"
echo "  layer=${LAYER} pc=${PC}"
echo "  self_starts=${SELF_STARTS}"
echo "  other_starts=${OTHER_STARTS}"
if [[ "${WINDOW_MODE}" == "fixed" ]]; then
  echo "  lengths=${LENGTHS}"
else
  echo "  self_end_shifts=${SELF_END_SHIFTS}"
  echo "  other_end_shifts=${OTHER_END_SHIFTS}"
fi
echo "  log=${LOG}"

nohup "${PYTHON}" -u scripts/run_fixed_window_broad_sweep.py \
  --patient "${PATIENT}" \
  --region "${REGION}" \
  --model "${MODEL}" \
  --context_tag "${CONTEXT_TAG}" \
  --layer "${LAYER}" \
  --pc "${PC}" \
  --window_mode "${WINDOW_MODE}" \
  --lengths="${LENGTHS}" \
  --self_starts="${SELF_STARTS}" \
  --other_starts="${OTHER_STARTS}" \
  --self_end_shifts="${SELF_END_SHIFTS}" \
  --other_end_shifts="${OTHER_END_SHIFTS}" \
  > "${LOG}" 2>&1 &

PID="$!"
echo "Started PID=${PID}"
echo "Follow with:"
echo "  tail -f ${LOG}"
