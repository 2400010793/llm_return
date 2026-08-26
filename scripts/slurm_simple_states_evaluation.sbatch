#!/bin/bash
#SBATCH --job-name=simple-states
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_simple_states/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_simple_states/%A_%a.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?simple_states array must use scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?run as an array job}"
: "${MANIFEST:?export MANIFEST}"

DATA_ROOT=${DATA_ROOT:-/data/alpha_team2/shares/simple_states_data}
SIMPLE_STATES_ROOT=${SIMPLE_STATES_ROOT:-/mnt/lustre3/home/gaozh/alpha_team2_zengl}
EXPECTED_SIMPLE_STATES_COMMIT=${EXPECTED_SIMPLE_STATES_COMMIT:-d8ebaa0}
TOP_N=${TOP_N:-100}
MIN_COUNT=${MIN_COUNT:-20}
MIN_AMOUNT=${MIN_AMOUNT:-20000000}
mkdir -p logs/slurm_simple_states
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

actual_commit=$(git -C "${SIMPLE_STATES_ROOT}" rev-parse --short=7 HEAD)
[[ "${actual_commit}" == "${EXPECTED_SIMPLE_STATES_COMMIT:0:7}" ]] || {
  echo "simple_states commit mismatch: expected ${EXPECTED_SIMPLE_STATES_COMMIT}, got ${actual_commit}" >&2
  exit 2
}

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id source_task_id predictions factor_id output_dir prediction_column duplicate_policy <<< "${line}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || {
  echo "manifest task mismatch" >&2
  exit 2
}
prediction_column=${prediction_column:-prediction}
duplicate_policy=${duplicate_policy:-error}

/usr/bin/python3 scripts/evaluate_predictions_simple_states.py \
  "${predictions}" \
  --output-dir "${output_dir}" \
  --factor-id "${factor_id}" \
  --prediction-column "${prediction_column}" \
  --prediction-date-role execution \
  --duplicate-policy "${duplicate_policy}" \
  --direction 1 --label-lag 1 --top-n "${TOP_N}" \
  --min-count "${MIN_COUNT}" --min-amount "${MIN_AMOUNT}" \
  --max-missing-weight 0.05 --annualization 250 \
  --data-root "${DATA_ROOT}" \
  --simple-states-root "${SIMPLE_STATES_ROOT}"
