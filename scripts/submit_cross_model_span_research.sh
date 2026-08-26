#!/bin/bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
MANIFEST=${MANIFEST:-${REPO_ROOT}/configs/generated/cross_model_span_manifest.tsv}
OUTPUT_ROOT=${OUTPUT_ROOT:-${SINA_ROOT}/prompt_minimal_v2/cross_model_span_v3}
mkdir -p "${REPO_ROOT}/logs/cross_model_span" "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"
.venv/bin/python -m py_compile scripts/run_cross_model_span_regression.py scripts/build_cross_model_span_manifest.py
bash -n scripts/slurm_cross_model_span_regression.sbatch
.venv/bin/python scripts/build_cross_model_span_manifest.py --output "${MANIFEST}" --include-raw
tasks=$(($(wc -l < "${MANIFEST}") - 1))
if [[ "${tasks}" -le 0 ]]; then exit 2; fi
job=$(sbatch --parsable --array="0-$((tasks - 1))%24" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},MANIFEST=${MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT}" \
  scripts/slurm_cross_model_span_regression.sbatch)
summary=$(sbatch --parsable --dependency="afterok:${job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}" \
  scripts/slurm_cross_model_span_summary.sbatch)
echo "cross_model_job=${job} summary_job=${summary} tasks=${tasks}"
