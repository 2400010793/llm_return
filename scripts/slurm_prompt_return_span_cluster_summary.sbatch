#!/bin/bash
#SBATCH --job-name=return-span-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/return_span_cluster/%j_summary.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/return_span_cluster/%j_summary.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/return_span_cluster_v1}
cd "${REPO_ROOT}"
.venv/bin/python scripts/summarize_prompt_return_span_clusters.py \
  --fold-root "${STUDY_ROOT}/folds" --output-root "${STUDY_ROOT}/summary" --strict
