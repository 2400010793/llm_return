#!/bin/bash
#SBATCH --job-name=xmodel-span-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/cross_model_span/%j_summary.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/cross_model_span/%j_summary.err
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-${SINA_ROOT}/prompt_minimal_v2/cross_model_span_v3}
cd "${REPO_ROOT}"
.venv/bin/python scripts/summarize_cross_model_span.py --root "${OUTPUT_ROOT}" --output "${OUTPUT_ROOT}/summary" --expected 648
