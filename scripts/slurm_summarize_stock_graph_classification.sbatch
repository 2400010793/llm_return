#!/bin/bash
#SBATCH --job-name=stock-graph-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_stock_graph/%j_summary.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_stock_graph/%j_summary.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
REPORT_ROOT=${REPORT_ROOT:-reports/classification/stock_graph_roberta_masked_v1}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}
/usr/bin/python3 scripts/summarize_stock_graph_classification.py \
  --input-dir "${REPORT_ROOT}" \
  --output "${REPORT_ROOT}/summary.json"
