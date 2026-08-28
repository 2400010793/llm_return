#!/bin/bash
# Record and submit the fair 4-model x 5-prompt x 5-feature x 5-classifier study.
# It reuses existing pooled embeddings and does not generate new embeddings.
set -euo pipefail
cd /home/team/llm_return

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name sina-cninfo-method-embedding-classifier-matrix-v1 \
    --purpose "Compare 4 existing embedding models across 5 prompt variants, 5 common pooled representations, and 5 classifiers on the aligned Sina panel" \
    --input /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/classification/sina_single_stock_classification_panel.parquet \
    --input /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/embeddings \
      --output /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/prompt_embedding_classification \
    --output configs/generated/sina_cninfo_method_embedding_classifier_v1.tsv \
    --related-file scripts/build_sina_cninfo_method_embedding_classifier_manifest.py \
    --related-file scripts/slurm_sina_cninfo_method_embedding_classifier.sbatch \
    --related-file scripts/run_pooled_embedding_classification.py \
    --related-file src/data/pooled_embeddings.py \
    --related-file src/evaluation/artifacts.py \
    --related-file src/models/dimension_reduction.py \
    --related-file src/evaluation/classification.py \
    --tag sina --tag cninfo-method --tag embedding-matrix \
    --tag prompt-variants --tag five-classifiers --tag leakage-safe --seed 42 \
    -- bash "$0" "$@"
fi

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

EMBEDDING_ROOT=${EMBEDDING_ROOT:-/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/embeddings}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/prompt_embedding_classification}
PANEL=${PANEL:-/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/classification/sina_single_stock_classification_panel.parquet}
MANIFEST=${MANIFEST:-configs/generated/sina_cninfo_method_embedding_classifier_v1.tsv}
EXPECTED_ROWS=${EXPECTED_ROWS:-4928}
EXPECTED_SHARDS=${EXPECTED_SHARDS:-4}
CLASSIFICATION_CONCURRENCY=${CLASSIFICATION_CONCURRENCY:-16}

mkdir -p configs/generated logs/slurm_sina_embedding_classifier

# Cheap local checks and the complete-asset manifest audit happen before sbatch.
.venv/bin/python -m py_compile \
  scripts/build_sina_cninfo_method_embedding_classifier_manifest.py \
  scripts/run_pooled_embedding_classification.py \
  src/data/pooled_embeddings.py \
  src/evaluation/artifacts.py \
  src/models/dimension_reduction.py \
  src/evaluation/classification.py
bash -n scripts/slurm_sina_cninfo_method_embedding_classifier.sbatch

.venv/bin/python scripts/build_sina_cninfo_method_embedding_classifier_manifest.py \
  --embedding-root "${EMBEDDING_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --panel "${PANEL}" \
  --manifest "${MANIFEST}" \
  --expected-shards "${EXPECTED_SHARDS}" \
  --expected-rows "${EXPECTED_ROWS}" \
  --reuse-output-root /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1 \
  --reuse-output-root /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/prompt_embedding_classification

tasks=$(($(wc -l < "${MANIFEST}") - 1))
if (( tasks < 1 || tasks > 500 )); then
  echo "Expected between 1 and 500 embedding classifier tasks, found ${tasks}" >&2
  exit 2
fi

job_id=$(tracked_sbatch --parsable \
  --array="0-$((tasks - 1))%${CLASSIFICATION_CONCURRENCY}" \
  --export="ALL,MANIFEST=${MANIFEST},PANEL=${PANEL}" \
  scripts/slurm_sina_cninfo_method_embedding_classifier.sbatch)
echo "embedding_classifier_matrix=${job_id} tasks=${tasks} concurrency=${CLASSIFICATION_CONCURRENCY}"
echo "manifest=${MANIFEST}"
