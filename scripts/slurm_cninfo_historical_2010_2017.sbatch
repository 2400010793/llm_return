#!/bin/bash
#SBATCH --job-name=cninfo-10-17
#SBATCH --partition=cpu
#SBATCH --array=0-815%20
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_cninfo_historical/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_cninfo_historical/%A_%a.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?historical CNINFO collection must use scripts/task_tracker.py}"

STOCK_INDEX=$((SLURM_ARRAY_TASK_ID + 1))
SOURCE_PANEL=${SOURCE_PANEL:-data/processed/cninfo_full_o2o_panel.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-data/interim/cninfo_paper_816_2010_2017}
RAW_DIR=${RAW_DIR:-data/raw/cninfo_paper_816_2010_2017}

mkdir -p logs/slurm_cninfo_historical "${OUTPUT_DIR}" "${RAW_DIR}"
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

# Annual shards make every completed year durable.  Rerunning this array is
# safe: complete stock-years are skipped and partial/error outputs are retried.
for YEAR in $(seq 2010 2017); do
  /usr/bin/python3 scripts/run_cninfo_paper_collection.py \
    --stocks data/stock_universe_paper_1000.csv \
    --stock-ids-from "${SOURCE_PANEL}" \
    --expected-stocks 816 \
    --start-index "${STOCK_INDEX}" \
    --end-index "${STOCK_INDEX}" \
    --start-date "${YEAR}-01-01" \
    --end-date "${YEAR}-12-31" \
    --limit 0 \
    --workers 1 \
    --stock-timeout-seconds 5400 \
    --max-pages 0 \
    --resolve-stock \
    --page-pause-seconds 2 \
    --pause-seconds 1 \
    --stock-pause-seconds 0 \
    --output-dir "${OUTPUT_DIR}" \
    --output-suffix "${YEAR}" \
    --raw-dir "${RAW_DIR}"
done
