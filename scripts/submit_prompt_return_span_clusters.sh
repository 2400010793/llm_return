#!/bin/bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
MECHANISM_ROOT=${MECHANISM_ROOT:-${SINA_ROOT}/prompt_minimal_v2/analysis_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/return_span_cluster_v1}
CACHE_MANIFEST=${CACHE_MANIFEST:-${REPO_ROOT}/configs/generated/minimal_prompt_cache_v2.tsv}
FOLD_MANIFEST=${FOLD_MANIFEST:-${REPO_ROOT}/configs/generated/minimal_prompt_knn_folds_v2.tsv}
CACHE_CONCURRENCY=${CACHE_CONCURRENCY:-4}
FOLD_CONCURRENCY=${FOLD_CONCURRENCY:-24}
cd "${REPO_ROOT}"
mkdir -p logs/return_span_cluster "${STUDY_ROOT}"
.venv/bin/python -m py_compile \
  scripts/build_prompt_return_span_cache.py scripts/run_prompt_return_span_cluster_fold.py \
  scripts/summarize_prompt_return_span_clusters.py
for script in scripts/slurm_prompt_return_span_{cache,cluster_fold,cluster_summary}.sbatch; do
  bash -n "${script}"
done
cache_job=$(sbatch --parsable --array="0-3%${CACHE_CONCURRENCY}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},STUDY_ROOT=${STUDY_ROOT},CACHE_MANIFEST=${CACHE_MANIFEST}" \
  scripts/slurm_prompt_return_span_cache.sbatch)
fold_job=$(sbatch --parsable --dependency="afterok:${cache_job}" --array="0-143%${FOLD_CONCURRENCY}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},MECHANISM_ROOT=${MECHANISM_ROOT},STUDY_ROOT=${STUDY_ROOT},FOLD_MANIFEST=${FOLD_MANIFEST}" \
  scripts/slurm_prompt_return_span_cluster_fold.sbatch)
summary_job=$(sbatch --parsable --dependency="afterok:${fold_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},STUDY_ROOT=${STUDY_ROOT}" \
  scripts/slurm_prompt_return_span_cluster_summary.sbatch)
echo "cache=${cache_job} folds=${fold_job} summary=${summary_job}"
