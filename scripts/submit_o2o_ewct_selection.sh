#!/bin/bash
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return

MANIFEST=${MANIFEST:-configs/generated/o2o_ewct_validation.tsv}
SELECTION_OUTPUT=${SELECTION_OUTPUT:-reports/strategy/o2o_ewct_validation/selection.json}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name o2o-ewct-selection \
    --purpose "Select EWCT gamma and quantile count on 2024-2025 only, then apply the frozen configuration to the disclosed 2026 confirmation sample" \
    --input "${MANIFEST}" --input data/processed/cninfo_full_o2o_market.parquet \
    --output reports/strategy/o2o_ewct_validation \
    --output reports/strategy/o2o_ewct_2026_confirmation.json \
    --related-file scripts/select_o2o_ewct.py \
    --related-file scripts/run_selected_o2o_ewct.py \
    --related-file scripts/slurm_o2o_ewct_validation.sbatch \
    --related-file scripts/slurm_o2o_ewct_select.sbatch \
    --related-file scripts/slurm_o2o_ewct_confirm.sbatch \
    --tag o2o --tag ewct --tag validation-selection --tag backtest --seed 42 \
    -- bash "$0" "$@"
fi

tasks=$(( $(wc -l < "${MANIFEST}") - 1 ))
[[ "${tasks}" -eq 8 ]] || { echo "expected eight validation cells; found ${tasks}" >&2; exit 2; }
mkdir -p logs/slurm_o2o_ewct reports/strategy/o2o_ewct_validation

validation_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --array="0-$((tasks - 1))%8" \
      --export="ALL,MANIFEST=${MANIFEST}" scripts/slurm_o2o_ewct_validation.sbatch
)
echo "submitted validation=${validation_job} tasks=${tasks}"

selection_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency="afterok:${validation_job}" \
      --export="ALL,MANIFEST=${MANIFEST},SELECTION_OUTPUT=${SELECTION_OUTPUT}" \
      scripts/slurm_o2o_ewct_select.sbatch
)
echo "submitted selection=${selection_job} dependency=${validation_job}"

confirmation_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency="afterok:${selection_job}" \
      --export="ALL,SELECTION_OUTPUT=${SELECTION_OUTPUT}" \
      scripts/slurm_o2o_ewct_confirm.sbatch
)
echo "submitted confirmation=${confirmation_job} dependency=${selection_job}"
