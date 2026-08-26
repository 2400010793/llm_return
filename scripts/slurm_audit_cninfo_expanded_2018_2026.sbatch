#!/bin/bash
#SBATCH --job-name=cninfo-18-26-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_cninfo_expanded_2018_2026_audit/%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_cninfo_expanded_2018_2026_audit/%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?expanded CNINFO audit must use scripts/task_tracker.py}"
: "${STOCKS_FILE:?STOCKS_FILE is required}"
: "${EXPECTED_STOCKS:?EXPECTED_STOCKS is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${FINAL_END_DATE:?FINAL_END_DATE is required}"
: "${AUDIT_OUTPUT:?AUDIT_OUTPUT is required}"
mkdir -p logs/slurm_cninfo_expanded_2018_2026_audit reports

"${REPO_ROOT}/.venv/bin/python" scripts/audit_cninfo_annual_collection.py \
  --stocks "${STOCKS_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --start-year 2018 \
  --end-year 2026 \
  --final-end-date "${FINAL_END_DATE}" \
  --expected-stocks "${EXPECTED_STOCKS}" \
  --summary "${AUDIT_OUTPUT}"
