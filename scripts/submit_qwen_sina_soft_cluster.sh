#!/bin/bash
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return

REGRESSION_ROOT=${REGRESSION_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/qwen3_prompt_return_regression_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REGRESSION_ROOT}/soft_cluster_v1}
PANEL=${PANEL:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/classification/sina_single_stock_classification_panel.parquet}
FOLD_MANIFEST=${FOLD_MANIFEST:-${OUTPUT_ROOT}/fold_manifest.tsv}
CONFIG_MANIFEST=${CONFIG_MANIFEST:-${OUTPUT_ROOT}/config_manifest.tsv}
CONCURRENCY=${CONCURRENCY:-54}
SNAPSHOT_FROM_TASK=${SNAPSHOT_FROM_TASK:-}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  tracker_args=()
  [[ -n "${SNAPSHOT_FROM_TASK}" ]] && tracker_args+=(--snapshot-from-task "${SNAPSHOT_FROM_TASK}")
  exec .venv/bin/python scripts/task_tracker.py run \
    --name qwen-sina-return-soft-cluster-v1 \
    --purpose "Leakage-safe Qwen Sina return-prompt soft clustering for article mean, prompt mean, and return token" \
    --input "${REGRESSION_ROOT}" --input "${PANEL}" \
    --output "${OUTPUT_ROOT}" \
    --related-file scripts/build_qwen_sina_soft_cluster_manifest.py \
    --related-file scripts/run_positive_token_soft_cluster_fold.py \
    --related-file scripts/slurm_positive_token_soft_cluster_fold.sbatch \
    --related-file scripts/slurm_positive_token_soft_cluster_evaluate.sbatch \
    --related-file scripts/summarize_positive_token_soft_cluster.py \
    --tag qwen --tag sina --tag soft-cluster --tag regression --seed 42 \
    "${tracker_args[@]}" -- bash "$0" "$@"
fi

mkdir -p "${OUTPUT_ROOT}" logs/positive_token_soft_cluster
.venv/bin/python scripts/build_qwen_sina_soft_cluster_manifest.py \
  --regression-root "${REGRESSION_ROOT}" --panel "${PANEL}" \
  --fold-manifest "${FOLD_MANIFEST}" --config-manifest "${CONFIG_MANIFEST}"

fold_tasks=$(( $(wc -l < "${FOLD_MANIFEST}") - 1 ))
config_tasks=$(( $(wc -l < "${CONFIG_MANIFEST}") - 1 ))
fold_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --array="0-$((fold_tasks - 1))%${CONCURRENCY}" \
      --export="ALL,MANIFEST=${FOLD_MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT}" \
      scripts/slurm_positive_token_soft_cluster_fold.sbatch
)
eval_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency="afterok:${fold_job}" \
      --array="0-$((config_tasks - 1))%6" \
      --export="ALL,CONFIG_MANIFEST=${CONFIG_MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT},GAMMAS=0.1" \
      scripts/slurm_positive_token_soft_cluster_evaluate.sbatch
)
summary_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency="afterok:${eval_job}" \
      --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT}" \
      scripts/slurm_positive_token_soft_cluster_summary.sbatch
)
echo "folds=${fold_job} evaluation=${eval_job} summary=${summary_job} fold_tasks=${fold_tasks} configs=${config_tasks}"
