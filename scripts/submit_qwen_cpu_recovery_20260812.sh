#!/bin/bash
set -euo pipefail

REPO_ROOT=/home/team/llm_return
cd "${REPO_ROOT}"

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name qwen-cpu-recovery-20260812 \
    --purpose "Recover pooled Qwen preparation and run classification/regression screens serially with 256 GiB per model task" \
    --input data/processed/embeddings/qwen3_embedding_8b_candidates_1000 \
    --input data/processed/cninfo_full_classification_panel.parquet \
    --input data/processed/cninfo_full_o2o_residual_panel.parquet \
    --output data/processed/pooled_embeddings_qwen_candidates_1000 \
    --output reports/classification/qwen_fullpanel_screen_20260811 \
    --output reports/classification/qwen_followup_screen_20260811 \
    --output reports/classification/qwen_o2o_screen_20260811 \
    --output reports/regression/qwen_fullpanel_screen_20260811 \
    --output reports/regression/qwen_followup_screen_20260811 \
    --related-file scripts/slurm_prepare_qwen_pooled_20260811.sbatch \
    --related-file scripts/slurm_qwen_classification_screen_20260811.sbatch \
    --related-file scripts/slurm_qwen_regression_screen_20260811.sbatch \
    --related-file configs/generated/qwen_overnight_classification_20260811.tsv \
    --related-file configs/generated/qwen_followup_classification_20260811.tsv \
    --related-file configs/generated/qwen_o2o_classification_20260811.tsv \
    --related-file configs/generated/qwen_overnight_regression_20260811.tsv \
    --related-file configs/generated/qwen_followup_regression_20260811.tsv \
    --tag ollama --tag qwen --tag recovery --tag memory-safe --seed 42 \
    -- bash "$0" "$@"
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
TRACKER=("${PYTHON}" scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" --)
PREPARE_SCRIPT=scripts/slurm_prepare_qwen_pooled_20260811.sbatch
CLASSIFICATION_SCRIPT=scripts/slurm_qwen_classification_screen_20260811.sbatch
REGRESSION_SCRIPT=scripts/slurm_qwen_regression_screen_20260811.sbatch

test -x "${PYTHON}"
test -d data/processed/embeddings/qwen3_embedding_8b_candidates_1000
test -s data/processed/cninfo_full_classification_panel.parquet
test -s data/processed/cninfo_full_o2o_residual_panel.parquet
bash -n "${PREPARE_SCRIPT}" "${CLASSIFICATION_SCRIPT}" "${REGRESSION_SCRIPT}"
"${PYTHON}" -m py_compile \
  scripts/prepare_qwen_pooled_candidates.py \
  scripts/run_pooled_embedding_classification.py \
  scripts/run_pooled_embedding_regression.py

submit_array() {
  local dependency=$1
  local manifest=$2
  local script=$3
  local tasks
  tasks=$(( $(wc -l < "${manifest}") - 1 ))
  (( tasks > 0 ))
  "${TRACKER[@]}" sbatch --parsable \
    --dependency="afterok:${dependency}" \
    --array="0-$((tasks - 1))%1" \
    --export=ALL,MANIFEST="${manifest}" \
    "${script}"
}

prepare_id=$("${TRACKER[@]}" sbatch --parsable "${PREPARE_SCRIPT}")
full_classification_id=$(submit_array "${prepare_id}" \
  configs/generated/qwen_overnight_classification_20260811.tsv \
  "${CLASSIFICATION_SCRIPT}")
followup_classification_id=$(submit_array "${full_classification_id}" \
  configs/generated/qwen_followup_classification_20260811.tsv \
  "${CLASSIFICATION_SCRIPT}")
o2o_classification_id=$(submit_array "${followup_classification_id}" \
  configs/generated/qwen_o2o_classification_20260811.tsv \
  "${CLASSIFICATION_SCRIPT}")
base_regression_id=$(submit_array "${o2o_classification_id}" \
  configs/generated/qwen_overnight_regression_20260811.tsv \
  "${REGRESSION_SCRIPT}")
followup_regression_id=$(submit_array "${base_regression_id}" \
  configs/generated/qwen_followup_regression_20260811.tsv \
  "${REGRESSION_SCRIPT}")

echo "submitted qwen_prepare=${prepare_id}"
echo "submitted qwen_full_classification=${full_classification_id} dependency=afterok:${prepare_id} throttle=1"
echo "submitted qwen_followup_classification=${followup_classification_id} dependency=afterok:${full_classification_id} throttle=1"
echo "submitted qwen_o2o_classification=${o2o_classification_id} dependency=afterok:${followup_classification_id} throttle=1"
echo "submitted qwen_base_regression=${base_regression_id} dependency=afterok:${o2o_classification_id} throttle=1"
echo "submitted qwen_followup_regression=${followup_regression_id} dependency=afterok:${base_regression_id} throttle=1"
