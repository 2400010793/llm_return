#!/bin/bash
set -euo pipefail
cd /home/team/llm_return

MANIFEST=${MANIFEST:-configs/generated/pooled_o2o_classification_seed42.tsv}
PANEL=${PANEL:-data/processed/cninfo_full_o2o_panel.parquet}
MAX_PARALLEL=${MAX_PARALLEL:-8}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name o2o-classification-core \
    --purpose "Train dedicated O2O direction classifiers with the 2026 fold held out for final testing" \
    --input "${PANEL}" \
    --input data/processed/pooled_embeddings_v3 \
    --output reports/classification/pooled_embeddings/o2o_core \
    --related-file scripts/build_o2o_classification_manifest.py \
    --related-file scripts/slurm_o2o_classification.sbatch \
    --related-file scripts/run_pooled_embedding_classification.py \
    --seed 42 \
    -- bash "$0" "$@"
fi

mkdir -p configs/generated logs/slurm_o2o_classification
test -s "${PANEL}"
.venv/bin/python scripts/build_o2o_classification_manifest.py --output "${MANIFEST}"
.venv/bin/python -m py_compile \
  scripts/build_o2o_classification_manifest.py \
  scripts/run_pooled_embedding_classification.py
bash -n scripts/slurm_o2o_classification.sbatch "$0"

tasks=$(( $(wc -l < "${MANIFEST}") - 1 ))
if [[ "${tasks}" -ne 96 ]]; then
  echo "Expected 96 O2O tasks, found ${tasks}" >&2
  exit 2
fi

job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --array="0-$((tasks - 1))%${MAX_PARALLEL}" \
      --export="ALL,MANIFEST=${MANIFEST},PANEL=${PANEL},FILTER_COLUMN=o2o_target_eligible" \
      scripts/slurm_o2o_classification.sbatch
)
echo "submitted O2O classification job=${job} tasks=${tasks} max_parallel=${MAX_PARALLEL}"
