#!/bin/bash
# Run the bounded paper-style one-step LSTM baseline on existing embeddings.
# This entrypoint never generates embeddings.
set -euo pipefail
cd /home/gaozh/llm_return

PANEL=${PANEL:-data/processed/cninfo_full_classification_panel.parquet}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-$PWD}
MANIFEST=${MANIFEST:-configs/generated/pooled_lstm_classification_seed42.tsv}
MODELS=${MODELS:-roberta}
VARIANTS=${VARIANTS:-short}
MAX_PARALLEL=${MAX_PARALLEL:-1}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name pooled-lstm-classification-v1 \
    --purpose "Run a bounded paper-style one-step LSTM baseline on existing pooled CNINFO embeddings; use next-day direction labels and do not generate embeddings." \
    --input "${PANEL}" \
    --input data/processed/pooled_embeddings_v3 \
    --input data/processed/pooled_long_embeddings_v4 \
    --output reports/classification/pooled_embeddings/lstm \
    --output "${MANIFEST}" \
    --related-file scripts/build_pooled_classification_manifest.py \
    --related-file scripts/run_pooled_embedding_classification.py \
    --related-file scripts/slurm_pooled_embedding_classification.sbatch \
    --related-file scripts/submit_pooled_lstm_classification.sh \
    --related-file src/models/lstm_classifier.py \
    --related-file src/models/representation_models.py \
    --related-file tests/test_lstm_classifier.py \
    --tag cninfo --tag pooled-embedding --tag lstm --tag next-day \
    --tag one-step --tag no-embeddings --seed 42 \
    -- bash "$0" "$@"
fi

mkdir -p configs/generated logs/slurm_pooled_cls reports/classification/pooled_embeddings/lstm
test -s "${PANEL}"
.venv/bin/python -m py_compile \
  scripts/build_pooled_classification_manifest.py \
  scripts/run_pooled_embedding_classification.py \
  src/models/lstm_classifier.py \
  src/models/representation_models.py
bash -n scripts/slurm_pooled_embedding_classification.sbatch "$0"

.venv/bin/python scripts/build_pooled_classification_manifest.py \
  --phase lstm \
  --root "${EMBEDDING_ROOT}" \
  --models "${MODELS}" \
  --variants "${VARIANTS}" \
  --output "${MANIFEST}"

tasks=$(($(wc -l < "${MANIFEST}") - 1))
if [[ "${tasks}" -lt 1 ]]; then
  echo "Expected at least one LSTM task, found ${tasks}" >&2
  exit 2
fi

job_id=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --array="0-$((tasks - 1))%${MAX_PARALLEL}" \
      --export="ALL,MANIFEST=${MANIFEST},PANEL=${PANEL}" \
      scripts/slurm_pooled_embedding_classification.sbatch
)
echo "pooled_lstm_classification=${job_id} tasks=${tasks} max_parallel=${MAX_PARALLEL}"
echo "manifest=${MANIFEST}"