#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

: "${MANIFEST:?export MANIFEST}"
: "${OUTPUT_ROOT:?export OUTPUT_ROOT}"
TASK_NAME=${TASK_NAME:-manifest-portfolio-backtest}
MAX_PARALLEL=${MAX_PARALLEL:-6}
SNAPSHOT_FROM_TASK=${SNAPSHOT_FROM_TASK:-}
MARKET=${MARKET:-data/processed/cninfo_full_o2o_market_2010_2026_hfq.parquet}
GAMMAS=${GAMMAS:-0.1}
GAMMAS_ENCODED=${GAMMAS//,/:}
PARTITION=${PARTITION:-${SLURM_JOB_PARTITION:-cpu}}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  tracker_args=()
  [[ -n "${SNAPSHOT_FROM_TASK}" ]] && tracker_args+=(--snapshot-from-task "${SNAPSHOT_FROM_TASK}")
  exec .venv/bin/python scripts/task_tracker.py run \
    --name "${TASK_NAME}" \
    --purpose "Run manifest stock-day predictions through existing 5bp long-only and long-short portfolio framework" \
    --input "${MANIFEST}" --input "${MARKET}" \
    --output "${OUTPUT_ROOT}" \
    --related-file scripts/submit_manifest_portfolio_backtest.sh \
    --related-file scripts/slurm_manifest_portfolio_backtest.sbatch \
    --related-file scripts/run_portfolio_strategy.py \
    --related-file src/portfolio/strategy.py \
    --tag regression --tag portfolio --tag 5bp --seed 42 \
    "${tracker_args[@]}" -- bash "$0" "$@"
fi

tasks=$(( $(wc -l < "${MANIFEST}") - 1 ))
[[ "${tasks}" -gt 0 ]] || { echo "manifest has no tasks" >&2; exit 2; }
mkdir -p logs/slurm_manifest_portfolio "${OUTPUT_ROOT}"

job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --job-name="${TASK_NAME}" --partition="${PARTITION}" \
      --array="0-$((tasks - 1))%${MAX_PARALLEL}" \
      --export="ALL,MANIFEST=${MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT},MARKET=${MARKET},GAMMAS_ENCODED=${GAMMAS_ENCODED}" \
      scripts/slurm_manifest_portfolio_backtest.sbatch
)
echo "submitted portfolio array=${job} tasks=${tasks} max_parallel=${MAX_PARALLEL} gammas=${GAMMAS}"
