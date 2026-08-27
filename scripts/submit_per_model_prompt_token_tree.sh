#!/bin/bash
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return

FAIR_ROOT=${FAIR_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/three_model_four_prompt_fair_pca_v2}
TREE_ROOT=${TREE_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/per_model_prompt_token_tree_v1}
REPORT=${REPORT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/REPORT_ALL_RESULTS.md}
BUILD_SINA_JOB=${BUILD_SINA_JOB:-5107846}
BUILD_CNINFO_JOB=${BUILD_CNINFO_JOB:-5107847}
CPU_EXCLUDE=${CPU_EXCLUDE:-c001-epyc9755,c003-epyc9755,c005-epyc9755,c006-epyc9755,c010-epyc9755,c011-epyc9755,c012-epyc9755,c118-epyc9575f,v123-epyc9575f,v126-epyc9575f,v127-epyc9575f,v128-epyc9575f,v129-epyc9575f,v130-epyc9575f,v131-epyc9575f,v132-epyc9575f,v133-epyc9575f,v134-epyc9575f,v135-epyc9575f,v136-epyc9575f,v139-epyc9575f}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name per-model-prompt-token-tree-v1 \
    --purpose "Independent tree models for each LLM using four prompt-token OOS factors; no cross-model fusion" \
    --input "${FAIR_ROOT}" --output "${TREE_ROOT}" --output "${REPORT}" \
    --related-file scripts/run_per_model_prompt_token_tree.py \
    --related-file scripts/slurm_per_model_prompt_token_tree.sbatch \
    --related-file scripts/summarize_per_model_prompt_token_tree.py \
    --tag cninfo --tag sina --tag roberta --tag bge-m3 --tag qwen --tag token --tag tree --tag independent --seed 42 \
    -- bash "$0" "$@"
fi

mkdir -p "${TREE_ROOT}" logs/per_model_prompt_tree
tracked() { .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- "$@"; }
rolling_sina=$(tracked sbatch --parsable --exclude="${CPU_EXCLUDE}" --dependency="afterok:${BUILD_SINA_JOB}" --array=0-215%24 \
  --export="ALL,OUTPUT_ROOT=${FAIR_ROOT}/sina,VALIDATION_YEARS=2" scripts/slurm_three_model_prompt_rolling.sbatch)
rolling_cninfo=$(tracked sbatch --parsable --exclude="${CPU_EXCLUDE}" --dependency="afterok:${BUILD_CNINFO_JOB}" --array=0-143%24 \
  --export="ALL,OUTPUT_ROOT=${FAIR_ROOT}/cninfo,VALIDATION_YEARS=1" scripts/slurm_three_model_prompt_rolling.sbatch)
tree_job=$(tracked sbatch --parsable --exclude="${CPU_EXCLUDE}" --dependency="afterok:${rolling_sina}:${rolling_cninfo}" --array=0-5%6 \
  --export="ALL,FAIR_ROOT=${FAIR_ROOT},TREE_ROOT=${TREE_ROOT}" scripts/slurm_per_model_prompt_token_tree.sbatch)
summary_job=$(tracked sbatch --parsable --partition=cpu --cpus-per-task=4 --mem=16G --time=02:00:00 \
  --exclude="${CPU_EXCLUDE}" --dependency="afterok:${tree_job}" \
  --export="ALL,TREE_ROOT=${TREE_ROOT},REPORT=${REPORT}" \
  --wrap=".venv/bin/python scripts/summarize_per_model_prompt_token_tree.py --tree-root '${TREE_ROOT}' --report '${REPORT}'")
echo "rolling_sina=${rolling_sina} rolling_cninfo=${rolling_cninfo} tree=${tree_job} summary=${summary_job}"
