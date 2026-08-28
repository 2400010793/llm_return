#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

ROOT=${ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/qwen3_prompt_return_regression_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/soft_cluster_v1}
PANEL=${PANEL:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/classification/sina_single_stock_classification_panel.parquet}
FOLD_MANIFEST=${FOLD_MANIFEST:-${OUTPUT_ROOT}/raw_fold_manifest.tsv}
CONFIG_MANIFEST=${CONFIG_MANIFEST:-${OUTPUT_ROOT}/raw_config_manifest.tsv}
SNAPSHOT_FROM_TASK=${SNAPSHOT_FROM_TASK:-}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  tracker_args=()
  [[ -n "${SNAPSHOT_FROM_TASK}" ]] && tracker_args+=(--snapshot-from-task "${SNAPSHOT_FROM_TASK}")
  exec .venv/bin/python scripts/task_tracker.py run \
    --name qwen-sina-return-raw-soft-cluster-v1 \
    --purpose "Direct raw-4096 versus PCA soft-cluster ablation for Qwen Sina return token" \
    --input "${ROOT}" --input "${PANEL}" --output "${OUTPUT_ROOT}" \
    --related-file scripts/build_qwen_sina_raw_soft_cluster_manifest.py \
    --related-file scripts/run_positive_token_soft_cluster_fold.py \
    --related-file scripts/slurm_positive_token_soft_cluster_fold.sbatch \
    --related-file scripts/slurm_positive_token_soft_cluster_evaluate.sbatch \
    --tag qwen --tag sina --tag raw-cluster --tag pca-ablation --seed 42 \
    "${tracker_args[@]}" -- bash "$0" "$@"
fi

.venv/bin/python scripts/build_qwen_sina_raw_soft_cluster_manifest.py \
  --regression-root "${ROOT}" --panel "${PANEL}" \
  --fold-manifest "${FOLD_MANIFEST}" --config-manifest "${CONFIG_MANIFEST}"
fold_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --cpus-per-task=16 --mem=96G --time=1-00:00:00 \
      --array="0-17%9" --export="ALL,MANIFEST=${FOLD_MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT}" \
      scripts/slurm_positive_token_soft_cluster_fold.sbatch
)
eval_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency="afterok:${fold_job}" --array="0-1%2" \
      --export="ALL,CONFIG_MANIFEST=${CONFIG_MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT},GAMMAS=0.1" \
      scripts/slurm_positive_token_soft_cluster_evaluate.sbatch
)
summary_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency="afterok:${eval_job}" \
      --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT}" scripts/slurm_positive_token_soft_cluster_summary.sbatch
)
echo "raw_folds=${fold_job} evaluation=${eval_job} summary=${summary_job}"
