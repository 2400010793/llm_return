#!/bin/bash
set -euo pipefail

REPO_ROOT=/home/team/llm_return
cd "${REPO_ROOT}"

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name qwen-failed-screens-20260812 \
    --purpose "Recover failed strict O2O classification and blocked regression screens with one 256 GiB task at a time" \
    --input data/processed/pooled_embeddings_qwen_candidates_1000 \
    --input data/processed/cninfo_full_o2o_panel.parquet \
    --input data/processed/cninfo_full_o2o_residual_panel.parquet \
    --output reports/classification/qwen_o2o_screen_20260811 \
    --output reports/regression/qwen_fullpanel_screen_20260811 \
    --output reports/regression/qwen_followup_screen_20260811 \
    --related-file scripts/slurm_qwen_classification_screen_20260811.sbatch \
    --related-file scripts/slurm_qwen_regression_screen_20260811.sbatch \
    --related-file configs/generated/qwen_o2o_classification_20260811.tsv \
    --related-file configs/generated/qwen_overnight_regression_20260811.tsv \
    --related-file configs/generated/qwen_followup_regression_20260811.tsv \
    --tag ollama --tag qwen --tag recovery --tag memory-safe --seed 42 \
    -- bash "$0" "$@"
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
TRACKER=("${PYTHON}" scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" --)
CLASSIFICATION_SCRIPT=scripts/slurm_qwen_classification_screen_20260811.sbatch
REGRESSION_SCRIPT=scripts/slurm_qwen_regression_screen_20260811.sbatch
O2O_PANEL=data/processed/cninfo_full_o2o_panel.parquet

test -x "${PYTHON}"
test -d data/processed/pooled_embeddings_qwen_candidates_1000
test -s "${O2O_PANEL}"
test -s data/processed/cninfo_full_o2o_residual_panel.parquet
bash -n "${CLASSIFICATION_SCRIPT}" "${REGRESSION_SCRIPT}"
"${PYTHON}" -m py_compile \
  scripts/run_pooled_embedding_classification.py \
  scripts/run_pooled_embedding_regression.py

submit_array() {
  local dependency=$1
  local manifest=$2
  local script=$3
  shift 3
  local tasks
  tasks=$(( $(wc -l < "${manifest}") - 1 ))
  (( tasks > 0 ))
  "${TRACKER[@]}" sbatch --parsable \
    --dependency="${dependency}" \
    --array="0-$((tasks - 1))%1" \
    --export="ALL,MANIFEST=${manifest}$*" \
    "${script}"
}

o2o_classification_id=$("${TRACKER[@]}" sbatch --parsable \
  --array="0-20%1" \
  --export="ALL,MANIFEST=configs/generated/qwen_o2o_classification_20260811.tsv,PANEL=${O2O_PANEL}" \
  "${CLASSIFICATION_SCRIPT}")

base_regression_id=$(submit_array "afterany:${o2o_classification_id}" \
  configs/generated/qwen_overnight_regression_20260811.tsv \
  "${REGRESSION_SCRIPT}")

followup_regression_id=$(submit_array "afterany:${base_regression_id}" \
  configs/generated/qwen_followup_regression_20260811.tsv \
  "${REGRESSION_SCRIPT}")

echo "submitted qwen_o2o_classification=${o2o_classification_id} tasks=21 throttle=1 panel=${O2O_PANEL}"
echo "submitted qwen_base_regression=${base_regression_id} tasks=6 throttle=1 dependency=afterany:${o2o_classification_id}"
echo "submitted qwen_followup_regression=${followup_regression_id} tasks=15 throttle=1 dependency=afterany:${base_regression_id}"
