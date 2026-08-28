#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
DATASET_ROOT=${DATASET_ROOT:-data/processed/stock_graph_features_roberta_masked_v1}
REPORT_ROOT=${REPORT_ROOT:-reports/classification/stock_graph_roberta_masked_v1}
MANIFEST=${MANIFEST:-configs/generated/stock_graph_roberta_masked_v1.tsv}
MAX_PARALLEL=${MAX_PARALLEL:-6}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name stock-graph-roberta-masked-v1 \
    --purpose "Build stock-day nodes from masked RoBERTa title/body embeddings and compare self-only, industry GraphSAGE, and degree-preserving random-graph controls under rolling next-day classification" \
    --input "${PANEL}" \
    --input data/processed/pooled_embeddings_2010_2026_roberta_v1 \
    --input data/stock_universe_paper_1000.csv \
    --output "${DATASET_ROOT}" \
    --output "${REPORT_ROOT}" \
    --output "${MANIFEST}" \
    --related-file scripts/build_stock_graph_features.py \
    --related-file scripts/build_stock_graph_manifest.py \
    --related-file scripts/run_stock_graph_rolling_classification.py \
    --related-file scripts/summarize_stock_graph_classification.py \
    --related-file src/models/stock_graph_sage.py \
    --related-file scripts/slurm_build_stock_graph_features.sbatch \
    --related-file scripts/slurm_stock_graph_rolling_classification.sbatch \
    --related-file scripts/slurm_summarize_stock_graph_classification.sbatch \
    --related-file scripts/submit_stock_graph_roberta_masked_v1.sh \
    --related-file tests/test_stock_graph_features.py \
    --related-file tests/test_stock_graph_sage.py \
    --tag cninfo --tag graph --tag graphsage --tag roberta --tag masked \
    --tag next-day --tag cpu-only --seed 42 \
    -- bash "$0" "$@"
fi

mkdir -p configs/generated logs/slurm_stock_graph "${REPORT_ROOT}"
test -s "${PANEL}"
test -s data/stock_universe_paper_1000.csv
for script in \
  scripts/slurm_build_stock_graph_features.sbatch \
  scripts/slurm_stock_graph_rolling_classification.sbatch \
  scripts/slurm_summarize_stock_graph_classification.sbatch \
  scripts/submit_stock_graph_roberta_masked_v1.sh; do
  bash -n "${script}"
done
.venv/bin/python -m py_compile \
  scripts/build_stock_graph_features.py \
  scripts/build_stock_graph_manifest.py \
  scripts/run_stock_graph_rolling_classification.py \
  scripts/summarize_stock_graph_classification.py \
  src/models/stock_graph_sage.py
.venv/bin/python scripts/build_stock_graph_manifest.py \
  --output "${MANIFEST}" \
  --report-root "${REPORT_ROOT}"
tasks=$(($(wc -l < "${MANIFEST}") - 1))
if [[ "${tasks}" -ne 27 ]]; then
  echo "Expected 27 graph tasks, found ${tasks}" >&2
  exit 2
fi

build_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable \
      --export="ALL,PANEL=${PANEL},OUTPUT_ROOT=${DATASET_ROOT}" \
      scripts/slurm_build_stock_graph_features.sbatch
)
classification_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency="afterok:${build_job}" \
      --array="0-$((tasks - 1))%${MAX_PARALLEL}" \
      --export="ALL,MANIFEST=${MANIFEST},DATASET_ROOT=${DATASET_ROOT}" \
      scripts/slurm_stock_graph_rolling_classification.sbatch
)
summary_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency="afterok:${classification_job}" \
      --export="ALL,REPORT_ROOT=${REPORT_ROOT}" \
      scripts/slurm_summarize_stock_graph_classification.sbatch
)

echo "stock_graph_build=${build_job} memory=96G cpu_only=true"
echo "stock_graph_classification=${classification_job} tasks=${tasks} max_parallel=${MAX_PARALLEL} memory=64G dependency=afterok:${build_job}"
echo "stock_graph_summary=${summary_job} dependency=afterok:${classification_job}"
