#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

SOURCE_PANEL=${SOURCE_PANEL:-data/processed/cninfo_full_o2o_panel.parquet}
MARKET_DATA=${MARKET_DATA:-data/processed/cninfo_full_o2o_market.parquet}
RESIDUAL_PANEL=${RESIDUAL_PANEL:-data/processed/cninfo_full_o2o_residual_panel.parquet}
MAIN_MANIFEST=${MAIN_MANIFEST:-configs/generated/o2o_residual_refinement_main.tsv}
ZSCORE_MANIFEST=${ZSCORE_MANIFEST:-configs/generated/o2o_residual_refinement_zscore.tsv}
ALPHAS=${ALPHAS:-0.00001,0.0001,0.001,0.01,0.1,1,10,50,100,1000,10000}
ALPHAS_ENCODED=${ALPHAS//,/;}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name o2o-residual-refinement \
    --purpose "Compare raw, market-residual, winsor-residual and cross-sectional standardized O2O targets with robust PCA regressors" \
    --input "${SOURCE_PANEL}" --input "${MARKET_DATA}" \
    --input data/processed/pooled_embeddings_v3 \
    --output "${RESIDUAL_PANEL}" \
    --output reports/regression/pooled_embeddings/stock_day \
    --related-file src/data/o2o_residual_targets.py \
    --related-file scripts/build_o2o_residual_panel.py \
    --related-file scripts/build_pooled_regression_manifest.py \
    --related-file scripts/slurm_build_o2o_residual_panel.sbatch \
    --related-file scripts/slurm_pooled_embedding_regression.sbatch \
    --tag o2o --tag regression --tag residual-target --seed 42 \
    -- bash "$0" "$@"
fi

mkdir -p configs/generated logs/slurm_o2o_residual logs/slurm_pooled_regression
.venv/bin/python -m py_compile \
  src/data/o2o_residual_targets.py scripts/build_o2o_residual_panel.py

build_job=""
if [[ ! -s "${RESIDUAL_PANEL}" || ! -s "${RESIDUAL_PANEL%.parquet}.summary.json" ]]; then
  build_job=$(
    .venv/bin/python scripts/task_tracker.py child-submit \
      --task-id "${TASK_RECORD_ID}" -- \
      sbatch --parsable \
        --export="ALL,SOURCE_PANEL=${SOURCE_PANEL},MARKET_DATA=${MARKET_DATA},OUTPUT_PANEL=${RESIDUAL_PANEL}" \
        scripts/slurm_build_o2o_residual_panel.sbatch
  )
  echo "submitted residual panel build=${build_job}"
else
  echo "using existing residual panel=${RESIDUAL_PANEL}"
fi

.venv/bin/python scripts/build_pooled_regression_manifest.py \
  --model roberta --variants masked_short --feature title_body_full_concat \
  --targets next_day_open_to_open_return,next_day_open_to_open_market_residual,next_day_open_to_open_winsor_residual \
  --regressors ridge,huber_sgd --run-mode screen \
  --reducers pca:32,pca:64,pca:128,pca:256 --expected-rows 350577 \
  --output "${MAIN_MANIFEST}"

.venv/bin/python scripts/build_pooled_regression_manifest.py \
  --model roberta --variants masked_short --feature title_body_full_concat \
  --targets next_day_open_to_open_cs_zscore \
  --regressors ridge --run-mode screen \
  --reducers pca:32,pca:64,pca:128,pca:256 --expected-rows 350577 \
  --output "${ZSCORE_MANIFEST}"

dependency_args=()
if [[ -n "${build_job}" ]]; then
  dependency_args+=(--dependency="afterok:${build_job}")
fi

submit_matrix() {
  local name=$1 manifest=$2 throttle=$3
  local tasks
  tasks=$(( $(wc -l < "${manifest}") - 1 ))
  [[ "${tasks}" -gt 0 ]] || { echo "empty manifest ${manifest}" >&2; exit 2; }
  local job
  job=$(
    .venv/bin/python scripts/task_tracker.py child-submit \
      --task-id "${TASK_RECORD_ID}" -- \
      sbatch --parsable "${dependency_args[@]}" \
        --job-name="${name}" --array="0-$((tasks - 1))%${throttle}" \
        --export="ALL,MANIFEST=${manifest},PANEL=${RESIDUAL_PANEL},SEARCH_STAGE=fine,ALPHAS_ENCODED=${ALPHAS_ENCODED},MAX_ALPHA_EXPANSIONS=2" \
        scripts/slurm_pooled_embedding_regression.sbatch
  )
  echo "submitted ${name}=${job} tasks=${tasks} dependency=${build_job:-none}"
}

submit_matrix o2o-resid-main "${MAIN_MANIFEST}" 4
submit_matrix o2o-resid-zscore "${ZSCORE_MANIFEST}" 4
