#!/bin/bash
#SBATCH --job-name=cninfo-18-26-all
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_cninfo_expanded_2018_2026/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_cninfo_expanded_2018_2026/%A_%a.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?expanded CNINFO collection must use scripts/task_tracker.py}"
: "${STOCKS_FILE:?STOCKS_FILE is required}"
: "${EXPECTED_STOCKS:?EXPECTED_STOCKS is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${RAW_DIR:?RAW_DIR is required}"
: "${FINAL_END_DATE:?FINAL_END_DATE is required}"

STOCK_INDEX_OFFSET=${STOCK_INDEX_OFFSET:-0}
STOCK_INDEX=$((STOCK_INDEX_OFFSET + SLURM_ARRAY_TASK_ID + 1))
mkdir -p logs/slurm_cninfo_expanded_2018_2026 "${OUTPUT_DIR}" "${RAW_DIR}"
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

failures=0
for YEAR in $(seq 2018 2026); do
  END_DATE="${YEAR}-12-31"
  [[ "${YEAR}" == "2026" ]] && END_DATE="${FINAL_END_DATE}"
  /usr/bin/python3 scripts/run_cninfo_paper_collection.py \
    --stocks "${STOCKS_FILE}" \
    --expected-stocks "${EXPECTED_STOCKS}" \
    --start-index "${STOCK_INDEX}" \
    --end-index "${STOCK_INDEX}" \
    --start-date "${YEAR}-01-01" \
    --end-date "${END_DATE}" \
    --limit 0 \
    --workers 1 \
    --stock-timeout-seconds 5400 \
    --max-pages 0 \
    --resolve-stock \
    --page-pause-seconds 0.1 \
    --pause-seconds 0.1 \
    --stock-pause-seconds 0 \
    --output-dir "${OUTPUT_DIR}" \
    --output-suffix "${YEAR}" \
    --raw-dir "${RAW_DIR}" || failures=$((failures + 1))
done
(( failures == 0 )) || exit 1
