#!/usr/bin/env bash
set -euo pipefail

# Submit one fixed- or variable-window configuration per SLURM array task.
# Environment-variable overrides are exported to every task.

PROJECT="${PROJECT:-/scratch/aniluchavez/hippocampal-speaker-semantics}"
WINDOW_MODE="${WINDOW_MODE:-fixed}"
LENGTHS="${LENGTHS:-200,300,500}"
SELF_STARTS="${SELF_STARTS:--500,-400,-300,-200,-100,0}"
OTHER_STARTS="${OTHER_STARTS:-0,20,100,200}"
SELF_END_SHIFTS="${SELF_END_SHIFTS:-0}"
OTHER_END_SHIFTS="${OTHER_END_SHIFTS:-100}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
PATIENTS="${PATIENTS:-${PATIENT:-PTYEU_task147}}"
DEPENDENCY="${DEPENDENCY:-}"

if [[ "${WINDOW_MODE}" != "fixed" && "${WINDOW_MODE}" != "varwin" ]]; then
    echo "WINDOW_MODE must be fixed or varwin" >&2
    exit 2
fi

count_csv() {
    local value="$1"
    local stripped="${value//[^,]/}"
    echo $((${#stripped} + 1))
}

n_self=$(count_csv "${SELF_STARTS}")
n_other=$(count_csv "${OTHER_STARTS}")
n_patients=$(count_csv "${PATIENTS}")
if [[ "${WINDOW_MODE}" == "fixed" ]]; then
    n_lengths=$(count_csv "${LENGTHS}")
    configs_per_patient=$((n_lengths * n_self * n_other))
else
    n_self_ends=$(count_csv "${SELF_END_SHIFTS}")
    n_other_ends=$(count_csv "${OTHER_END_SHIFTS}")
    configs_per_patient=$((n_self * n_other * n_self_ends * n_other_ends))
fi
total=$((n_patients * configs_per_patient))

mkdir -p "${PROJECT}/logs/window_sweep"

echo "Submitting ${total} ${WINDOW_MODE} tasks: ${n_patients} patients x ${configs_per_patient} windows (${MAX_CONCURRENT} concurrent)"
echo "SLURM will place tasks on available nodes in the guppy partition."

export WINDOW_MODE LENGTHS SELF_STARTS OTHER_STARTS SELF_END_SHIFTS OTHER_END_SHIFTS PATIENTS
export PROJECT
export REGION="${REGION:-hippocampus}"
export MODEL="${MODEL:-gpt2-large}"
export CONTEXT_TAG="${CONTEXT_TAG:-_ctx200}"
export LAYER="${LAYER:-36}"
export PC="${PC:-50}"

sbatch_args=(
    --array="0-$((total - 1))%${MAX_CONCURRENT}" \
    --export=ALL \
)
if [[ -n "${DEPENDENCY}" ]]; then
    sbatch_args+=(--dependency="${DEPENDENCY}")
fi
sbatch "${sbatch_args[@]}" "${PROJECT}/scripts/run_window_broad_sweep_array.sbatch"
