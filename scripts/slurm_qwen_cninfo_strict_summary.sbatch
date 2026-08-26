#!/bin/bash
#SBATCH --job-name=qwen-cninfo-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/qwen_cninfo_strict/%j.summary.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/qwen_cninfo_strict/%j.summary.err

set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
: "${REGRESSION_ROOT:?export REGRESSION_ROOT}"
: "${GEOMETRY_ROOT:?export GEOMETRY_ROOT}"
: "${SUMMARY_ROOT:?export SUMMARY_ROOT}"
mkdir -p "${SUMMARY_ROOT}" logs/qwen_cninfo_strict

.venv/bin/python scripts/summarize_qwen_cninfo_strict_comparison.py \
  --regression-root "${REGRESSION_ROOT}" \
  --geometry-root "${GEOMETRY_ROOT}" \
  --output-dir "${SUMMARY_ROOT}" --top-k 20
