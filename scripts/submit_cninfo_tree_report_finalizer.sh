#!/bin/bash
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return

: "${AFTER_JOB:?export AFTER_JOB}"
RESULT_DIR=${RESULT_DIR:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/results/tree_stacking_cninfo_oos_v1}
MAIN_REPORT=${MAIN_REPORT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/REPORT_ALL_RESULTS.md}
CPU_EXCLUDE=${CPU_EXCLUDE:-c001-epyc9755,c003-epyc9755,c005-epyc9755,c006-epyc9755,c010-epyc9755,c011-epyc9755,c012-epyc9755,c118-epyc9575f,v123-epyc9575f,v126-epyc9575f,v127-epyc9575f,v128-epyc9575f,v129-epyc9575f,v130-epyc9575f,v131-epyc9575f,v132-epyc9575f,v133-epyc9575f,v134-epyc9575f,v135-epyc9575f,v136-epyc9575f,v139-epyc9575f}
SNAPSHOT_FROM_TASK=${SNAPSHOT_FROM_TASK:-20260826T011835Z-qwen-cninfo-strict-token-full-v1-2505748}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name cninfo-tree-stacking-report-finalizer-v1 \
    --purpose "Append completed strict CNINFO tree-stacking results to REPORT_ALL_RESULTS.md" \
    --input "${RESULT_DIR}" --output "${MAIN_REPORT}" \
    --related-file scripts/update_report_cninfo_tree_stacking.py \
    --related-file scripts/slurm_update_cninfo_tree_report.sbatch \
    --tag cninfo --tag report --tag tree-model \
    --snapshot-from-task "${SNAPSHOT_FROM_TASK}" -- bash "$0" "$@"
fi

job=$(
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --exclude="${CPU_EXCLUDE}" --dependency="afterok:${AFTER_JOB}" \
      --export="ALL,RESULT_DIR=${RESULT_DIR},MAIN_REPORT=${MAIN_REPORT}" \
      scripts/slurm_update_cninfo_tree_report.sbatch
)
echo "tree_report_finalizer=${job}"
