#!/bin/bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
DATASET=${DATASET:-sina}
if [[ "${DATASET}" == "cninfo" ]]; then
  STUDY_ROOT=${STUDY_ROOT:-${REPO_ROOT}/reports/cninfo_minimal_prompt_v2/analysis_v1}
  EMBEDDING_ROOT=${EMBEDDING_ROOT:-${REPO_ROOT}/data/processed/cninfo_minimal_prompt_v2/embeddings}
  PANEL=${PANEL:-${REPO_ROOT}/data/processed/cninfo_full_classification_panel_2010_2026.parquet}
  EXPECTED_ROWS=${EXPECTED_ROWS:-903665}
  PROMPT_LENGTHS=${PROMPT_LENGTHS:-short}
  EXPECTED_FOLDS=72
  SUBMIT_SIMPLE_STATES=${SUBMIT_SIMPLE_STATES:-1}
else
  STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/analysis_v1}
  EMBEDDING_ROOT=${EMBEDDING_ROOT:-${SINA_ROOT}/prompt_minimal_v2/embeddings}
  PANEL=${PANEL:-${SINA_ROOT}/classification/sina_single_stock_classification_panel.parquet}
  EXPECTED_ROWS=${EXPECTED_ROWS:-75894}
  PROMPT_LENGTHS=${PROMPT_LENGTHS:-short long}
  EXPECTED_FOLDS=144
  SUBMIT_SIMPLE_STATES=${SUBMIT_SIMPLE_STATES:-0}
fi
CACHE_MANIFEST=${CACHE_MANIFEST:-${REPO_ROOT}/configs/generated/${DATASET}_minimal_prompt_cache_v2.tsv}
FOLD_MANIFEST=${FOLD_MANIFEST:-${REPO_ROOT}/configs/generated/${DATASET}_minimal_prompt_knn_folds_v2.tsv}
CACHE_CONCURRENCY=${CACHE_CONCURRENCY:-4}
FOLD_CONCURRENCY=${FOLD_CONCURRENCY:-256}
UPSTREAM_JOB=${UPSTREAM_JOB:-}
cd "${REPO_ROOT}"

mkdir -p configs/generated logs/minimal_prompt_knn "${STUDY_ROOT}"
.venv/bin/python -m py_compile \
  src/analysis/prompt_token_mechanisms.py scripts/build_prompt_only_baselines.py \
  scripts/build_prompt_mechanism_cache.py scripts/run_minimal_prompt_knn_fold.py \
  scripts/summarize_minimal_prompt_knn_v2.py scripts/build_minimal_prompt_knn_manifests.py
for script in scripts/slurm_minimal_prompt_{baselines,cache,knn_fold,summary}_v2.sbatch; do
  bash -n "${script}"
done
prompt_length_args=()
for length in ${PROMPT_LENGTHS}; do
  prompt_length_args+=(--prompt-length "${length}")
done
.venv/bin/python scripts/build_minimal_prompt_knn_manifests.py \
  --cache-manifest "${CACHE_MANIFEST}" --fold-manifest "${FOLD_MANIFEST}" \
  "${prompt_length_args[@]}"

dependency_args=()
[[ -n "${UPSTREAM_JOB}" ]] && dependency_args+=(--dependency="afterok:${UPSTREAM_JOB}")
common="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},STUDY_ROOT=${STUDY_ROOT},EMBEDDING_ROOT=${EMBEDDING_ROOT},PANEL=${PANEL},EXPECTED_ROWS=${EXPECTED_ROWS},PROMPT_LENGTHS=${PROMPT_LENGTHS},EXPECTED_FOLDS=${EXPECTED_FOLDS},SUBMIT_SIMPLE_STATES=${SUBMIT_SIMPLE_STATES}"
baseline_job=$(sbatch --parsable "${dependency_args[@]}" \
  --export="${common}" \
  scripts/slurm_minimal_prompt_baselines_v2.sbatch)
cache_tasks=$(( $(wc -l < "${CACHE_MANIFEST}") - 1 ))
fold_tasks=$(( $(wc -l < "${FOLD_MANIFEST}") - 1 ))
cache_job=$(sbatch --parsable "${dependency_args[@]}" --array="0-$((cache_tasks - 1))%${CACHE_CONCURRENCY}" \
  --export="${common},CACHE_MANIFEST=${CACHE_MANIFEST}" \
  scripts/slurm_minimal_prompt_cache_v2.sbatch)
fold_job=$(sbatch --parsable --dependency="afterok:${baseline_job}:${cache_job}" \
  --array="0-$((fold_tasks - 1))%${FOLD_CONCURRENCY}" \
  --export="${common},FOLD_MANIFEST=${FOLD_MANIFEST}" \
  scripts/slurm_minimal_prompt_knn_fold_v2.sbatch)
summary_job=$(sbatch --parsable --dependency="afterok:${fold_job}" \
  --export="${common}" \
  scripts/slurm_minimal_prompt_summary_v2.sbatch)
echo "baseline=${baseline_job} cache=${cache_job} folds=${fold_job} summary=${summary_job}"
echo "cache_tasks=${cache_tasks} cache_concurrency=${CACHE_CONCURRENCY} fold_tasks=${fold_tasks} fold_concurrency=${FOLD_CONCURRENCY} dataset=${DATASET}"
echo "study_root=${STUDY_ROOT}"
