#!/bin/bash
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return
SOURCE_PANEL=${SOURCE_PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
MARKET_OUTPUT=${MARKET_OUTPUT:-data/processed/cninfo_full_o2o_market_2010_2026_hfq.parquet}
O2O_PANEL=${O2O_PANEL:-data/processed/cninfo_full_o2o_panel_2010_2026_hfq.parquet}
O2O_CACHE=${O2O_CACHE:-data/interim/o2o_training_ohlc_hfq_2010_2026}
if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name o2o-data-2010-2026-hfq \
    --purpose "Clean and rebuild strict O2O labels for the complete 2010-2026 classification panel" \
    --input "${SOURCE_PANEL}" --output "${MARKET_OUTPUT}" --output "${O2O_PANEL}" \
    --related-file scripts/collect_strategy_ohlc.py \
    --related-file scripts/build_o2o_training_panel.py \
    --related-file scripts/slurm_prepare_o2o_training_data_2010_2026.sbatch \
    -- bash "$0" "$@"
fi
mkdir -p logs/slurm_o2o_2010_2026
job=$(.venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
  sbatch --parsable \
  --export="ALL,SOURCE_PANEL=${SOURCE_PANEL},MARKET_OUTPUT=${MARKET_OUTPUT},O2O_PANEL=${O2O_PANEL},O2O_CACHE=${O2O_CACHE}" \
  scripts/slurm_prepare_o2o_training_data_2010_2026.sbatch)
echo "submitted historical O2O data job=${job}"
