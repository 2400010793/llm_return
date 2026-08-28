#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

MANIFEST=${MANIFEST:-configs/generated/o2o_paper_replication_final_2026.tsv}
GAMMAS=${GAMMAS:-1.0,0.7,0.5,0.4,0.3,0.2,0.1}
GAMMAS_ENCODED=${GAMMAS//,/:}
SNAPSHOT_FROM_TASK=${SNAPSHOT_FROM_TASK:-}
MARKET=${MARKET:-data/processed/cninfo_full_o2o_market_2010_2026_hfq.parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/strategy/o2o_5bp_ewct_hfq}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  tracker_args=()
  [[ -n "${SNAPSHOT_FROM_TASK}" ]] && tracker_args+=(--snapshot-from-task "${SNAPSHOT_FROM_TASK}")
  exec .venv/bin/python scripts/task_tracker.py run \
    --name o2o-5bp-ewct-backtest \
    --purpose "Re-evaluate sealed O2O predictions with symmetric 5bp buy/sell costs and EWCT turnover limits" \
    --input "${MANIFEST}" \
    --input "${MARKET}" \
    --output "${OUTPUT_ROOT}" \
    --related-file scripts/slurm_o2o_5bp_ewct_backtest.sbatch \
    --related-file scripts/run_portfolio_strategy.py \
    --related-file src/portfolio/strategy.py \
    --tag o2o --tag backtest --tag 5bp --tag ewct --seed 42 \
    "${tracker_args[@]}" \
    -- bash "$0" "$@"
fi

tasks=$(( $(wc -l < "${MANIFEST}") - 1 ))
[[ "${tasks}" -gt 0 ]] || { echo "manifest has no tasks" >&2; exit 2; }
mkdir -p logs/slurm_o2o_5bp_ewct "${OUTPUT_ROOT}"

job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --job-name=o2o-5bp-ewct --array="0-$((tasks - 1))%4" \
      --export="ALL,MANIFEST=${MANIFEST},GAMMAS_ENCODED=${GAMMAS_ENCODED},MARKET=${MARKET},OUTPUT_ROOT=${OUTPUT_ROOT}" \
      scripts/slurm_o2o_5bp_ewct_backtest.sbatch
)
echo "submitted 5bp EWCT backtest=${job} tasks=${tasks} gammas=${GAMMAS}"
