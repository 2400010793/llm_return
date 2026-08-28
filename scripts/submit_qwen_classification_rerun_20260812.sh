#!/bin/bash
set -euo pipefail
cd /home/team/llm_return

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name qwen-classification-rerun-20260812 \
    --purpose "Reuse existing Ollama/Qwen embeddings and rerun all full-panel, follow-up, and strict O2O classification screens" \
    --input data/processed/embeddings/qwen3_embedding_8b_candidates_1000 \
    --input data/processed/cninfo_full_classification_panel.parquet \
    --output data/processed/pooled_embeddings_qwen_candidates_1000 \
    --output reports/classification/qwen_fullpanel_screen_20260811 \
    --output reports/classification/qwen_followup_screen_20260811 \
    --output reports/classification/qwen_o2o_screen_20260811 \
    --related-file scripts/prepare_qwen_pooled_candidates.py \
    --related-file scripts/slurm_prepare_qwen_pooled_20260811.sbatch \
    --related-file scripts/slurm_qwen_classification_screen_20260811.sbatch \
    --related-file configs/generated/qwen_overnight_classification_20260811.tsv \
    --related-file configs/generated/qwen_followup_classification_20260811.tsv \
    --related-file configs/generated/qwen_o2o_classification_20260811.tsv \
    --tag ollama --tag qwen --tag classification --tag rerun --seed 42 \
    -- bash "$0" "$@"
fi

PREPARE_SCRIPT=scripts/slurm_prepare_qwen_pooled_20260811.sbatch
CLASSIFICATION_SCRIPT=scripts/slurm_qwen_classification_screen_20260811.sbatch
MANIFESTS=(
  configs/generated/qwen_overnight_classification_20260811.tsv
  configs/generated/qwen_followup_classification_20260811.tsv
  configs/generated/qwen_o2o_classification_20260811.tsv
)
THROTTLES=(1 1 1)
LABELS=(fullpanel followup o2o)

test -x .venv/bin/python
test -s data/processed/cninfo_full_classification_panel.parquet
test -d data/processed/embeddings/qwen3_embedding_8b_candidates_1000
.venv/bin/python -m py_compile \
  scripts/prepare_qwen_pooled_candidates.py \
  scripts/run_pooled_embedding_classification.py
bash -n "${PREPARE_SCRIPT}"
bash -n "${CLASSIFICATION_SCRIPT}"

prepare_id=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable "${PREPARE_SCRIPT}"
)
echo "submitted qwen_pool_prepare=${prepare_id} using existing embeddings"

for index in "${!MANIFESTS[@]}"; do
  manifest=${MANIFESTS[$index]}
  test -s "${manifest}"
  tasks=$(( $(wc -l < "${manifest}") - 1 ))
  if (( tasks < 1 )); then
    echo "No tasks in ${manifest}" >&2
    exit 2
  fi
  classification_id=$(
    .venv/bin/python scripts/task_tracker.py child-submit \
      --task-id "${TASK_RECORD_ID}" -- \
      sbatch --parsable --dependency="afterok:${prepare_id}" \
        --array="0-$((tasks - 1))%${THROTTLES[$index]}" \
        --export=ALL,MANIFEST="${manifest}" \
        "${CLASSIFICATION_SCRIPT}"
  )
  echo "submitted qwen_${LABELS[$index]}_classification=${classification_id} tasks=${tasks} throttle=${THROTTLES[$index]} dependency=afterok:${prepare_id}"
done
