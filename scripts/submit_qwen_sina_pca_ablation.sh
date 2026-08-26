#!/bin/bash
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return

ROOT=${ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/qwen3_prompt_return_regression_v1}
PANEL=${PANEL:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/classification/sina_single_stock_classification_panel.parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/pca_ablation_v1}
MANIFEST=${MANIFEST:-${OUTPUT_ROOT}/manifest.tsv}
CONCURRENCY=${CONCURRENCY:-15}
SNAPSHOT_FROM_TASK=${SNAPSHOT_FROM_TASK:-}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  tracker_args=()
  [[ -n "${SNAPSHOT_FROM_TASK}" ]] && tracker_args+=(--snapshot-from-task "${SNAPSHOT_FROM_TASK}")
  exec .venv/bin/python scripts/task_tracker.py run \
    --name qwen-sina-pca-ablation-v1 \
    --purpose "Compare Qwen Sina Ridge without PCA against PCA32,64,128,256 on six aligned return-prompt representations" \
    --input "${ROOT}" --input "${PANEL}" --output "${OUTPUT_ROOT}" \
    --related-file scripts/build_qwen_sina_pca_ablation_manifest.py \
    --related-file scripts/slurm_qwen_sina_pca_ablation.sbatch \
    --related-file scripts/run_sina_precomputed_regression.py \
    --tag qwen --tag sina --tag pca-ablation --tag ridge --seed 42 \
    "${tracker_args[@]}" -- bash "$0" "$@"
fi

mkdir -p "${OUTPUT_ROOT}" logs/qwen_sina_pca
.venv/bin/python scripts/build_qwen_sina_pca_ablation_manifest.py \
  --regression-root "${ROOT}" --panel "${PANEL}" --output "${MANIFEST}"
tasks=$(( $(wc -l < "${MANIFEST}") - 1 ))
job=$(
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --array="0-$((tasks - 1))%${CONCURRENCY}" \
      --export="ALL,MANIFEST=${MANIFEST}" scripts/slurm_qwen_sina_pca_ablation.sbatch
)
echo "pca_ablation=${job} tasks=${tasks} concurrency=${CONCURRENCY}"
