#!/bin/bash
# Submit the audited 75,894-row Sina panel under two non-mixed target protocols.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
cd "${REPO_ROOT}"

SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
PANEL=${PANEL:-${SINA_ROOT}/classification/sina_single_stock_classification_panel.parquet}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-${SINA_ROOT}/embeddings}
OUTPUT_ROOT=${OUTPUT_ROOT:-${SINA_ROOT}/classification_results_v1}
MANIFEST=${MANIFEST:-configs/generated/sina_single_stock_classification_v1.tsv}
EXPECTED_ROWS=${EXPECTED_ROWS:-75894}
EXPECTED_SHARDS=${EXPECTED_SHARDS:-64}
CLASSIFICATION_CONCURRENCY=${CLASSIFICATION_CONCURRENCY:-16}
TASKS_EXPECTED=1000
SNAPSHOT_FROM_TASK=${SNAPSHOT_FROM_TASK:-20260818T071358Z-sina-single-stock-cninfo-v1-3961990}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name sina-single-stock-classification-v1 \
    --purpose "Compare paper sentiment event3-to-event3 classification with actionable next1-to-next1 classification on the audited Sina embeddings" \
    --input "${PANEL}" --input "${EMBEDDING_ROOT}" \
    --output "${OUTPUT_ROOT}" --output "${MANIFEST}" \
    --related-file scripts/build_sina_cninfo_method_embedding_classifier_manifest.py \
    --related-file scripts/slurm_sina_cninfo_method_embedding_classifier.sbatch \
    --related-file scripts/run_pooled_embedding_classification.py \
    --related-file src/data/pooled_embeddings.py \
    --related-file src/evaluation/artifacts.py \
    --tag sina --tag classification --tag event3-to-event3 \
    --tag next1-to-next1 --tag leakage-safe --seed 42 \
    --snapshot-from-task "${SNAPSHOT_FROM_TASK}" \
    -- bash "$0" "$@"
fi

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

mkdir -p configs/generated logs/slurm_sina_embedding_classifier "${OUTPUT_ROOT}"
.venv/bin/python -m py_compile \
  scripts/build_sina_cninfo_method_embedding_classifier_manifest.py \
  scripts/run_pooled_embedding_classification.py \
  src/data/pooled_embeddings.py
bash -n scripts/slurm_sina_cninfo_method_embedding_classifier.sbatch

.venv/bin/python scripts/build_sina_cninfo_method_embedding_classifier_manifest.py \
  --embedding-root "${EMBEDDING_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --panel "${PANEL}" \
  --manifest "${MANIFEST}" \
  --expected-shards "${EXPECTED_SHARDS}" \
  --expected-rows "${EXPECTED_ROWS}" \
  --target-mode event3_to_event3 \
  --target-mode next1_to_next1

tasks=$(($(wc -l < "${MANIFEST}") - 1))
if (( tasks != TASKS_EXPECTED )); then
  echo "Expected ${TASKS_EXPECTED} classification tasks, found ${tasks}" >&2
  exit 2
fi

job_id=$(tracked_sbatch --parsable \
  --array="0-$((tasks - 1))%${CLASSIFICATION_CONCURRENCY}" \
  --cpus-per-task=4 --mem=32G --time=2-00:00:00 \
  --export="ALL,REPO_ROOT=${REPO_ROOT},MANIFEST=${MANIFEST},PANEL=${PANEL},EXPECTED_ROWS=${EXPECTED_ROWS},MAX_MATRIX_GIB=4" \
  scripts/slurm_sina_cninfo_method_embedding_classifier.sbatch)

echo "sina_classification=${job_id} tasks=${tasks} concurrency=${CLASSIFICATION_CONCURRENCY}"
echo "event3_to_event3_tasks=500 next1_to_next1_tasks=500"
echo "manifest=${MANIFEST}"
