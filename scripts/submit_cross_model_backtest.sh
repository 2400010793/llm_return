#!/bin/bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
SUMMARY_ROOT=${SUMMARY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/cross_model_span_v3/summary}
cd "${REPO_ROOT}"
test -s "${SUMMARY_ROOT}/simple_states_manifest.tsv"
tasks=$(($(wc -l < "${SUMMARY_ROOT}/simple_states_manifest.tsv") - 1))
sbatch --parsable --array="0-$((tasks - 1))%18" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},SUMMARY_ROOT=${SUMMARY_ROOT}" \
  scripts/slurm_cross_model_simple_states.sbatch
