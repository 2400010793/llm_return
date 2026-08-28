#!/bin/bash
#SBATCH --job-name=cninfo-increment
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_cninfo_incremental/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_cninfo_incremental/%A_%a.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?incremental CNINFO collection must use scripts/task_tracker.py}"
: "${START_DATE:?START_DATE is required}"
: "${END_DATE:?END_DATE is required}"
: "${OUTPUT_SUFFIX:?OUTPUT_SUFFIX is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${RAW_DIR:?RAW_DIR is required}"
: "${STOCKS_FILE:?STOCKS_FILE is required}"
: "${EXPECTED_STOCKS:?EXPECTED_STOCKS is required}"

STOCK_INDEX_OFFSET=${STOCK_INDEX_OFFSET:-0}
STOCK_INDEX=$((STOCK_INDEX_OFFSET + SLURM_ARRAY_TASK_ID + 1))
mkdir -p logs/slurm_cninfo_incremental "${OUTPUT_DIR}" "${RAW_DIR}"
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

/usr/bin/python3 scripts/run_cninfo_paper_collection.py \
  --stocks "${STOCKS_FILE}" \
  --expected-stocks "${EXPECTED_STOCKS}" \
  --start-index "${STOCK_INDEX}" \
  --end-index "${STOCK_INDEX}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --limit 0 \
  --workers 1 \
  --stock-timeout-seconds 3600 \
  --max-pages 0 \
  --resolve-stock \
  --page-pause-seconds 2 \
  --pause-seconds 1 \
  --stock-pause-seconds 0 \
  --output-dir "${OUTPUT_DIR}" \
  --output-suffix "${OUTPUT_SUFFIX}" \
  --raw-dir "${RAW_DIR}"
