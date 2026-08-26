#!/bin/bash
#SBATCH --job-name=cninfo-incr-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_cninfo_incremental_audit/%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_cninfo_incremental_audit/%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?incremental CNINFO audit must use scripts/task_tracker.py}"
: "${START_DATE:?START_DATE is required}"
: "${END_DATE:?END_DATE is required}"
: "${OUTPUT_SUFFIX:?OUTPUT_SUFFIX is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${AUDIT_OUTPUT:?AUDIT_OUTPUT is required}"
: "${STOCKS_FILE:?STOCKS_FILE is required}"
: "${EXPECTED_STOCKS:?EXPECTED_STOCKS is required}"
mkdir -p logs/slurm_cninfo_incremental_audit reports

"${REPO_ROOT}/.venv/bin/python" scripts/audit_cninfo_incremental_collection.py \
  --stocks "${STOCKS_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --suffix "${OUTPUT_SUFFIX}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --expected-stocks "${EXPECTED_STOCKS}" \
  --summary "${AUDIT_OUTPUT}"
