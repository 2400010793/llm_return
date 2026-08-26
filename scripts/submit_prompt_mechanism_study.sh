#!/bin/bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_mechanism_v1}
CACHE_MANIFEST=${CACHE_MANIFEST:-${REPO_ROOT}/configs/generated/prompt_mechanism_cache_v1.tsv}
FOLD_MANIFEST=${FOLD_MANIFEST:-${REPO_ROOT}/configs/generated/prompt_mechanism_folds_v1.tsv}
CACHE_CONCURRENCY=${CACHE_CONCURRENCY:-4}
FOLD_CONCURRENCY=${FOLD_CONCURRENCY:-128}
cd "${REPO_ROOT}"

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name prompt-token-mechanism-study-v1 \
    --purpose "Test contextual prompt-token shifts, long-prompt semantic groups, stable clustering, and rolling Accuracy" \
    --input "${SINA_ROOT}/classification/sina_single_stock_classification_panel.parquet" \
    --input "${SINA_ROOT}/embeddings" --output "${STUDY_ROOT}" \
    --related-file src/analysis/prompt_token_mechanisms.py \
    --related-file scripts/build_prompt_mechanism_cache.py \
    --related-file scripts/run_prompt_mechanism_fold.py \
    --related-file scripts/summarize_prompt_mechanism_study.py \
    --tag prompt-token --tag mechanism --tag clustering --tag leakage-safe --seed 42 \
    -- bash "$0" "$@"
fi

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

mkdir -p configs/generated logs/prompt_mechanism "${STUDY_ROOT}"
.venv/bin/python -m py_compile \
  src/analysis/prompt_token_mechanisms.py scripts/build_prompt_only_baselines.py \
  scripts/build_prompt_mechanism_cache.py scripts/run_prompt_mechanism_fold.py \
  scripts/summarize_prompt_mechanism_study.py
for script in scripts/slurm_prompt_mechanism_{baselines,cache,fold,summary}.sbatch; do
  bash -n "${script}"
done
.venv/bin/python scripts/build_prompt_mechanism_manifests.py \
  --cache-manifest "${CACHE_MANIFEST}" --fold-manifest "${FOLD_MANIFEST}"

baseline_job=$(tracked_sbatch --parsable \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},STUDY_ROOT=${STUDY_ROOT}" \
  scripts/slurm_prompt_mechanism_baselines.sbatch)
cache_job=$(tracked_sbatch --parsable --array="0-7%${CACHE_CONCURRENCY}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},STUDY_ROOT=${STUDY_ROOT},CACHE_MANIFEST=${CACHE_MANIFEST}" \
  scripts/slurm_prompt_mechanism_cache.sbatch)
fold_job=$(tracked_sbatch --parsable --dependency="afterok:${baseline_job}:${cache_job}" \
  --array="0-287%${FOLD_CONCURRENCY}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},STUDY_ROOT=${STUDY_ROOT},FOLD_MANIFEST=${FOLD_MANIFEST}" \
  scripts/slurm_prompt_mechanism_fold.sbatch)
summary_job=$(tracked_sbatch --parsable --dependency="afterok:${fold_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},STUDY_ROOT=${STUDY_ROOT}" \
  scripts/slurm_prompt_mechanism_summary.sbatch)
echo "baseline=${baseline_job} cache=${cache_job} folds=${fold_job} summary=${summary_job}"
echo "cache_tasks=8 cache_concurrency=${CACHE_CONCURRENCY} fold_tasks=288 fold_concurrency=${FOLD_CONCURRENCY}"
echo "study_root=${STUDY_ROOT}"
