#!/bin/bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/return_span_logistic_v1}
cd "${REPO_ROOT}"
mkdir -p logs/return_span_logistic "${STUDY_ROOT}"
.venv/bin/python -m py_compile \
  scripts/run_return_span_logistic_fold.py scripts/summarize_return_span_logistic.py
bash -n scripts/slurm_return_span_logistic_fold.sbatch \
  scripts/slurm_return_span_logistic_summary.sbatch
fold_job=$(sbatch --parsable --array=0-26%27 \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},OUTPUT_ROOT=${STUDY_ROOT}/folds" \
  scripts/slurm_return_span_logistic_fold.sbatch)
summary_job=$(sbatch --parsable --dependency="afterok:${fold_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},STUDY_ROOT=${STUDY_ROOT}" \
  scripts/slurm_return_span_logistic_summary.sbatch)
echo "folds=${fold_job} summary=${summary_job}"
