#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

INPUT_ROOT=${INPUT_ROOT:-/data/alpha_team2/shares/llm_return/reports/regression/pooled_embeddings/o2o_rolling_2010_2026_hfq}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/results/tree_stacking_cninfo_oos_v1}
CPU_EXCLUDE=${CPU_EXCLUDE:-c001-epyc9755,c003-epyc9755,c005-epyc9755,c006-epyc9755,c010-epyc9755,c011-epyc9755,c012-epyc9755,c118-epyc9575f,v123-epyc9575f,v126-epyc9575f,v127-epyc9575f,v128-epyc9575f,v129-epyc9575f,v130-epyc9575f,v131-epyc9575f,v132-epyc9575f,v133-epyc9575f,v134-epyc9575f,v135-epyc9575f,v136-epyc9575f,v139-epyc9575f}
SNAPSHOT_FROM_TASK=${SNAPSHOT_FROM_TASK:-20260826T011835Z-qwen-cninfo-strict-token-full-v1-2505748}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name cninfo-roberta-bge-oos-tree-stacking-v1 \
    --purpose "Strict 6+2+1 second-level XGBoost LightGBM CatBoost aggregation of 72 CNINFO RoBERTa/BGE-M3 OOS factors" \
    --input "${INPUT_ROOT}" --output "${OUTPUT_ROOT}" \
    --related-file scripts/run_cninfo_oos_tree_stacking.py \
    --related-file scripts/slurm_cninfo_oos_tree_stacking.sbatch \
    --tag cninfo --tag roberta --tag bge-m3 --tag stacking --tag tree-model \
    --seed 42 --snapshot-from-task "${SNAPSHOT_FROM_TASK}" -- bash "$0" "$@"
fi

mkdir -p "${OUTPUT_ROOT}" logs/cninfo_tree_stacking
job=$(
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --exclude="${CPU_EXCLUDE}" \
      --export="ALL,INPUT_ROOT=${INPUT_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}" \
      scripts/slurm_cninfo_oos_tree_stacking.sbatch
)
echo "tree_stacking=${job}"
