#!/bin/bash
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return

MANIFEST=${MANIFEST:-configs/generated/o2o_final_2026.tsv}
PANEL=${PANEL:-data/processed/cninfo_full_o2o_residual_panel.parquet}
ALPHAS=${ALPHAS:-0.00001,0.0001,0.001,0.01,0.1,1,10,50,100,1000,10000}
ALPHAS_ENCODED=${ALPHAS//,/;}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name o2o-final-2026 \
    --purpose "Sealed 2026 OOS comparison and cost-aware backtest for four validation-selected O2O regressors" \
    --input "${PANEL}" --input data/processed/cninfo_full_o2o_market.parquet \
    --input data/processed/pooled_embeddings_v3 \
    --output reports/regression/pooled_embeddings/o2o_final_2026 \
    --output reports/strategy/o2o_final_2026 \
    --related-file "${MANIFEST}" \
    --related-file scripts/slurm_pooled_embedding_regression.sbatch \
    --related-file scripts/slurm_o2o_final_backtest.sbatch \
    --tag o2o --tag regression --tag final-oos --tag backtest --seed 42 \
    -- bash "$0" "$@"
fi

tasks=$(( $(wc -l < "${MANIFEST}") - 1 ))
[[ "${tasks}" -eq 4 ]] || { echo "expected four final candidates; found ${tasks}" >&2; exit 2; }
mkdir -p logs/slurm_o2o_final reports/regression/pooled_embeddings/o2o_final_2026

regression_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --job-name=o2o-final-2026 --array="0-$((tasks - 1))%4" \
      --export="ALL,MANIFEST=${MANIFEST},PANEL=${PANEL},SEARCH_STAGE=fine,ALPHAS_ENCODED=${ALPHAS_ENCODED},MAX_ALPHA_EXPANSIONS=2" \
      scripts/slurm_pooled_embedding_regression.sbatch
)
echo "submitted regression=${regression_job} tasks=${tasks}"

backtest_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --job-name=o2o-final-bt \
      --dependency="afterok:${regression_job}" --array="0-$((tasks - 1))%4" \
      --export="ALL,MANIFEST=${MANIFEST}" scripts/slurm_o2o_final_backtest.sbatch
)
echo "submitted backtest=${backtest_job} tasks=${tasks} dependency=${regression_job}"
