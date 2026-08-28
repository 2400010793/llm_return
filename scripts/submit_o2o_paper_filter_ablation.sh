#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

MANIFEST=${MANIFEST:-configs/generated/o2o_paper_filter_ablation.tsv}
if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name o2o-paper-filter-ablation \
    --purpose "Compare paper length filtering and rolling within-stock semantic novelty thresholds using the fixed winsor-residual Huber PCA-256 O2O model" \
    --input data/processed/cninfo_full_o2o_residual_panel.parquet \
    --input data/processed/cninfo_full_o2o_market.parquet \
    --input data/processed/pooled_embeddings_v3 \
    --output data/processed/paper_filters \
    --output reports/regression/pooled_embeddings/o2o_paper_filters \
    --related-file "${MANIFEST}" \
    --related-file src/data/rolling_semantic_dedup.py \
    --related-file scripts/build_cninfo_paper_filters.py \
    --related-file scripts/slurm_build_cninfo_paper_filters.sbatch \
    --related-file scripts/slurm_o2o_paper_filter_regression.sbatch \
    --tag o2o --tag paper-filter --tag semantic-dedup --tag regression --seed 42 \
    -- bash "$0" "$@"
fi

tasks=$(( $(wc -l < "${MANIFEST}") - 1 ))
[[ "${tasks}" -eq 5 ]] || { echo "expected five filter cells; found ${tasks}" >&2; exit 2; }
mkdir -p logs/slurm_paper_filters reports/regression/pooled_embeddings/o2o_paper_filters

build_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable scripts/slurm_build_cninfo_paper_filters.sbatch
)
echo "submitted build=${build_job}"

regression_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency="afterok:${build_job}" --array="0-$((tasks - 1))%5" \
      --export="ALL,MANIFEST=${MANIFEST}" scripts/slurm_o2o_paper_filter_regression.sbatch
)
echo "submitted regression=${regression_job} tasks=${tasks} dependency=${build_job}"
