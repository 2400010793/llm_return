#!/bin/bash
# Prepare historical O2O labels and submit validation-only regression screening.
set -euo pipefail
cd /home/team/llm_return

SOURCE_PANEL=${SOURCE_PANEL:-data/processed/cninfo_full_classification_panel.parquet}
MARKET_OUTPUT=${MARKET_OUTPUT:-data/processed/cninfo_full_o2o_market.parquet}
O2O_PANEL=${O2O_PANEL:-data/processed/cninfo_full_o2o_panel.parquet}
O2O_CACHE=${O2O_CACHE:-data/interim/o2o_training_ohlc_qfq}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name o2o-regression-screen \
    --purpose "Build strict historical O2O labels, then screen O2O return models without accessing the 2026 test fold" \
    --input "${SOURCE_PANEL}" \
    --output "${MARKET_OUTPUT}" \
    --output "${O2O_PANEL}" \
    --output reports/regression/pooled_embeddings/stock_day \
    --related-file scripts/slurm_prepare_o2o_training_data.sbatch \
    --related-file scripts/build_o2o_training_panel.py \
    --related-file scripts/build_pooled_regression_manifest.py \
    --related-file scripts/slurm_pooled_embedding_regression.sbatch \
    -- bash "$0" "$@"
fi

mkdir -p configs/generated logs/slurm_o2o logs/slurm_pooled_regression

data_job=""
if [[ ! -s "${O2O_PANEL}" ]]; then
  data_job=$(
    .venv/bin/python scripts/task_tracker.py child-submit \
      --task-id "${TASK_RECORD_ID}" -- \
      sbatch --parsable \
        --export="ALL,SOURCE_PANEL=${SOURCE_PANEL},MARKET_OUTPUT=${MARKET_OUTPUT},O2O_PANEL=${O2O_PANEL},O2O_CACHE=${O2O_CACHE}" \
        scripts/slurm_prepare_o2o_training_data.sbatch
  )
  echo "submitted O2O data preparation job=${data_job}"
else
  echo "using existing O2O panel=${O2O_PANEL}"
fi

submit_model() {
  local model=$1
  local manifest="configs/generated/o2o_regression_screen_${model}.tsv"
  .venv/bin/python scripts/build_pooled_regression_manifest.py \
    --model "${model}" \
    --variants short,masked_short \
    --feature title_body_full_concat \
    --targets next_day_open_to_open_return \
    --regressors ridge,elasticnet_sgd,huber_sgd,small_mlp \
    --run-mode screen \
    --reducers none,pca:128 \
    --expected-rows 350577 \
    --output "${manifest}"

  local tasks
  tasks=$(( $(wc -l < "${manifest}") - 1 ))
  [[ "${tasks}" -gt 0 ]] || { echo "empty manifest: ${manifest}" >&2; exit 2; }
  local dependency_args=()
  if [[ -n "${data_job}" ]]; then
    dependency_args+=(--dependency="afterok:${data_job}")
  fi
  local job
  job=$(
    .venv/bin/python scripts/task_tracker.py child-submit \
      --task-id "${TASK_RECORD_ID}" -- \
      sbatch --parsable "${dependency_args[@]}" --array="0-$((tasks - 1))%8" \
        --export="ALL,MANIFEST=${manifest},PANEL=${O2O_PANEL}" \
        scripts/slurm_pooled_embedding_regression.sbatch
  )
  echo "submitted ${model} O2O screen job=${job} tasks=${tasks} dependency=${data_job:-none}"
}

submit_model roberta
submit_model bge_m3
