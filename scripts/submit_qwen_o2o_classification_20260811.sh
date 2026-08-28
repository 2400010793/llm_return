#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name qwen-o2o-classification-20260811 \
    --purpose "Run strict Qwen O2O classification screens with identical open-to-open training and evaluation targets" \
    --input data/processed/pooled_embeddings_qwen_candidates_1000 \
    --input data/processed/cninfo_full_classification_panel.parquet \
    --output reports/classification/qwen_o2o_screen_20260811 \
    --related-file configs/generated/qwen_o2o_classification_20260811.tsv \
    --related-file scripts/slurm_qwen_classification_screen_20260811.sbatch \
    --tag qwen --tag o2o --tag classification --tag strict-target --seed 42 \
    -- bash "$0" "$@"
fi

PREPARE_JOB=4015405
MANIFEST=configs/generated/qwen_o2o_classification_20260811.tsv
OUTPUT_DIR=reports/classification/qwen_o2o_screen_20260811

test -s "${MANIFEST}"
bash -n scripts/slurm_qwen_classification_screen_20260811.sbatch
mkdir -p "${OUTPUT_DIR}"

tasks=$(( $(wc -l < "${MANIFEST}") - 1 ))
if (( tasks != 21 )); then
  echo "Expected 21 O2O classification tasks, found ${tasks}" >&2
  exit 2
fi

classification_id=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency=afterok:${PREPARE_JOB} \
      --array="0-$((tasks - 1))%2" \
      --export=ALL,MANIFEST="${MANIFEST}" \
      scripts/slurm_qwen_classification_screen_20260811.sbatch
)

echo "submitted qwen_o2o_classification=${classification_id} tasks=${tasks} throttle=2 dependency=afterok:${PREPARE_JOB}"
