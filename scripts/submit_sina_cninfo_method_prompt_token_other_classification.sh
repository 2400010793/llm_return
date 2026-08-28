#!/bin/bash
# Record and submit the additional Prompt-token classifier array. Existing
# Logistic Prompt-token and pooled jobs are not changed or resubmitted.
set -euo pipefail
cd /home/team/llm_return

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name sina-cninfo-method-prompt-token-other-classifiers-v1 \
    --purpose "Run Linear SVM and SGD classifiers on contextual Prompt-token PCA128 features" \
    --input /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/classification/sina_single_stock_classification_panel.parquet \
    --input /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/embeddings \
    --output /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/prompt_token_classification_other \
    --output configs/generated/sina_cninfo_method_prompt_token_other_classification_v1.tsv \
    --related-file scripts/build_sina_cninfo_method_prompt_token_other_classification_manifest.py \
    --related-file scripts/slurm_sina_cninfo_method_prompt_token_other_classification.sbatch \
    --related-file scripts/run_pooled_embedding_classification.py \
    --related-file src/data/pooled_embeddings.py \
    --related-file src/evaluation/artifacts.py \
    --related-file src/models/dimension_reduction.py \
    --tag sina --tag cninfo-method --tag prompt-token --tag classification \
    --tag linear-svm --tag sgd --tag leakage-safe --seed 42 \
    -- bash "$0" "$@"
fi

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

EMBEDDING_ROOT=${EMBEDDING_ROOT:-/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/embeddings}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1}
PANEL=${PANEL:-/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/classification/sina_single_stock_classification_panel.parquet}
MANIFEST=${MANIFEST:-configs/generated/sina_cninfo_method_prompt_token_other_classification_v1.tsv}
EXPECTED_ROWS=${EXPECTED_ROWS:-4928}
EXPECTED_SHARDS=${EXPECTED_SHARDS:-4}
CLASSIFICATION_CONCURRENCY=${CLASSIFICATION_CONCURRENCY:-2}

mkdir -p configs/generated logs/slurm_sina_cninfo_method

.venv/bin/python -m py_compile \
  scripts/build_sina_cninfo_method_prompt_token_other_classification_manifest.py \
  scripts/run_pooled_embedding_classification.py \
  src/data/pooled_embeddings.py \
  src/evaluation/artifacts.py
bash -n scripts/slurm_sina_cninfo_method_prompt_token_other_classification.sbatch

.venv/bin/python scripts/build_sina_cninfo_method_prompt_token_other_classification_manifest.py \
  --embedding-root "${EMBEDDING_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --manifest "${MANIFEST}" \
  --expected-shards "${EXPECTED_SHARDS}" \
  --expected-rows "${EXPECTED_ROWS}"

tasks=$(($(wc -l < "${MANIFEST}") - 1))
if [[ "${tasks}" != "32" ]]; then
  echo "Expected 32 additional Prompt-token tasks, found ${tasks}" >&2
  exit 2
fi

job_id=$(tracked_sbatch --parsable \
  --array="0-$((tasks - 1))%${CLASSIFICATION_CONCURRENCY}" \
  --export="ALL,MANIFEST=${MANIFEST},PANEL=${PANEL}" \
  scripts/slurm_sina_cninfo_method_prompt_token_other_classification.sbatch)
echo "prompt_token_other_classification=${job_id} tasks=${tasks} concurrency=${CLASSIFICATION_CONCURRENCY}"
echo "manifest=${MANIFEST}"
