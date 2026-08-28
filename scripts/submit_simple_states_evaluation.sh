#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

MANIFEST=${MANIFEST:-configs/generated/simple_states_finbert2_all_v1.tsv}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/simple_states/finbert2_all_v1}
TASK_NAME=${TASK_NAME:-simple-states-finbert2-all-v1}
MAX_PARALLEL=${MAX_PARALLEL:-8}
SNAPSHOT_FROM_TASK=${SNAPSHOT_FROM_TASK:-}
TOP_N=${TOP_N:-100}
MIN_COUNT=${MIN_COUNT:-20}
MIN_AMOUNT=${MIN_AMOUNT:-20000000}
DATA_ROOT=${DATA_ROOT:-/data/alpha_team2/shares/simple_states_data}
SIMPLE_STATES_ROOT=${SIMPLE_STATES_ROOT:-/mnt/lustre3/home/team/alpha_team2_zengl}
EXPECTED_SIMPLE_STATES_COMMIT=${EXPECTED_SIMPLE_STATES_COMMIT:-d8ebaa0}
PARTITION=${PARTITION:-${SLURM_JOB_PARTITION:-cpu}}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  tracker_args=()
  [[ -n "${SNAPSHOT_FROM_TASK}" ]] && tracker_args+=(--snapshot-from-task "${SNAPSHOT_FROM_TASK}")
  exec .venv/bin/python scripts/task_tracker.py run \
    --name "${TASK_NAME}" \
    --purpose "Evaluate completed stock-day regression factors with simple_states top_n=${TOP_N}, min_count=${MIN_COUNT}, min_amount=${MIN_AMOUNT}" \
    --input "${MANIFEST}" \
    --output "${OUTPUT_ROOT}" \
    --related-file scripts/submit_simple_states_evaluation.sh \
    --related-file scripts/slurm_simple_states_evaluation.sbatch \
    --related-file scripts/evaluate_predictions_simple_states.py \
    --tag regression --tag factor-evaluation --tag simple-states \
    "${tracker_args[@]}" \
    -- bash "$0" "$@"
fi

tasks=$(( $(wc -l < "${MANIFEST}") - 1 ))
[[ "${tasks}" -gt 0 ]] || { echo "manifest has no tasks" >&2; exit 2; }
[[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_PARALLEL must be positive" >&2; exit 2; }
[[ "${TOP_N}" =~ ^[1-9][0-9]*$ ]] || { echo "TOP_N must be positive" >&2; exit 2; }
[[ "${MIN_COUNT}" =~ ^[1-9][0-9]*$ ]] || { echo "MIN_COUNT must be positive" >&2; exit 2; }
mkdir -p logs/slurm_simple_states "${OUTPUT_ROOT}"

job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --job-name="${TASK_NAME}" --partition="${PARTITION}" \
      --array="0-$((tasks - 1))%${MAX_PARALLEL}" \
      --export="ALL,MANIFEST=${MANIFEST},TOP_N=${TOP_N},MIN_COUNT=${MIN_COUNT},MIN_AMOUNT=${MIN_AMOUNT},DATA_ROOT=${DATA_ROOT},SIMPLE_STATES_ROOT=${SIMPLE_STATES_ROOT},EXPECTED_SIMPLE_STATES_COMMIT=${EXPECTED_SIMPLE_STATES_COMMIT}" \
      scripts/slurm_simple_states_evaluation.sbatch
)
echo "submitted simple_states array=${job} tasks=${tasks} max_parallel=${MAX_PARALLEL} partition=${PARTITION} top_n=${TOP_N} min_count=${MIN_COUNT} min_amount=${MIN_AMOUNT}"
