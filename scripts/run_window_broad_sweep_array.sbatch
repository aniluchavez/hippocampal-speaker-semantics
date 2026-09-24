#!/usr/bin/env bash
#SBATCH --job-name=win_sweep
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/window_sweep/%A_%a.out
#SBATCH --error=/scratch/aniluchavez/hippocampal-speaker-semantics/logs/window_sweep/%A_%a.err

set -euo pipefail

PROJECT="${PROJECT:-/scratch/aniluchavez/hippocampal-speaker-semantics}"
PYTHON="${PYTHON:-/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3}"

PATIENTS="${PATIENTS:-${PATIENT:-PTYEU_task147}}"
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

IFS=',' read -r -a lengths <<< "${LENGTHS}"
IFS=',' read -r -a self_starts <<< "${SELF_STARTS}"
IFS=',' read -r -a other_starts <<< "${OTHER_STARTS}"
IFS=',' read -r -a self_ends <<< "${SELF_END_SHIFTS}"
IFS=',' read -r -a other_ends <<< "${OTHER_END_SHIFTS}"
IFS=',' read -r -a patients <<< "${PATIENTS}"

global_task="${SLURM_ARRAY_TASK_ID}"
if [[ "${WINDOW_MODE}" == "fixed" ]]; then
    n_other=${#other_starts[@]}
    n_self=${#self_starts[@]}
    configs_per_patient=$((${#lengths[@]} * n_self * n_other))
    patient_idx=$((global_task / configs_per_patient))
    task=$((global_task % configs_per_patient))
    other_idx=$((task % n_other))
    self_idx=$(((task / n_other) % n_self))
    length_idx=$((task / (n_other * n_self)))

    length="${lengths[$length_idx]}"
    self_start="${self_starts[$self_idx]}"
    other_start="${other_starts[$other_idx]}"
else
    n_other_end=${#other_ends[@]}
    n_self_end=${#self_ends[@]}
    n_other=${#other_starts[@]}
    configs_per_patient=$((${#self_starts[@]} * n_other * n_self_end * n_other_end))
    patient_idx=$((global_task / configs_per_patient))
    task=$((global_task % configs_per_patient))

    other_end_idx=$((task % n_other_end))
    self_end_idx=$(((task / n_other_end) % n_self_end))
    other_idx=$(((task / (n_other_end * n_self_end)) % n_other))
    self_idx=$((task / (n_other_end * n_self_end * n_other)))

    self_start="${self_starts[$self_idx]}"
    other_start="${other_starts[$other_idx]}"
    self_end="${self_ends[$self_end_idx]}"
    other_end="${other_ends[$other_end_idx]}"
fi

PATIENT="${patients[$patient_idx]}"
TASK_SUMMARY="/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/window_sweep_tasks/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}.csv"
mkdir -p "$(dirname "${TASK_SUMMARY}")"
cd "${PROJECT}"

echo "job=${SLURM_ARRAY_JOB_ID} global_task=${global_task} config_task=${task} patient=${PATIENT} host=$(hostname) started=$(date --iso-8601=seconds)"

common=(
    "${PYTHON}" -u scripts/run_fixed_window_broad_sweep.py
    --patient "${PATIENT}"
    --region "${REGION}"
    --model "${MODEL}"
    --context_tag "${CONTEXT_TAG}"
    --layer "${LAYER}"
    --pc "${PC}"
    --window_mode "${WINDOW_MODE}"
    --self_starts="${self_start}"
    --other_starts="${other_start}"
    --summary_path "${TASK_SUMMARY}"
)

if [[ "${WINDOW_MODE}" == "fixed" ]]; then
    "${common[@]}" --lengths="${length}"
else
    "${common[@]}" --self_end_shifts="${self_end}" --other_end_shifts="${other_end}"
fi

echo "completed=$(date --iso-8601=seconds) summary=${TASK_SUMMARY}"
