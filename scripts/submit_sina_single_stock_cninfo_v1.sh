#!/bin/bash
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name sina-single-stock-cninfo-v1 \
    --purpose "Rebuild Sina single-stock embeddings with the audited CNINFO prompt, mask, token, and pooling protocol" \
    --input /mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/cleaned_full_2010_2026/sina_all_news_clean_expanded.parquet \
    --input /mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/classification/sina_full_classification_panel.parquet \
    --output /mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1 \
    --related-file scripts/build_sina_single_stock_cninfo_dataset.py \
    --related-file scripts/build_sina_cninfo_method_inputs.py \
    --related-file scripts/run_sina_cninfo_method_embeddings.py \
    --related-file scripts/slurm_prepare_sina_single_stock_cninfo_v1.sbatch \
    --related-file scripts/slurm_sina_single_stock_cninfo_embeddings_v1.sbatch \
    --tag sina --tag single-stock --tag prompt-token --tag cninfo-method --tag cpu-safe \
    --seed 42 -- bash "$0" "$@"
fi

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

mkdir -p logs/sina_single_stock_cninfo_v1
prepare_job=$(tracked_sbatch --parsable scripts/slurm_prepare_sina_single_stock_cninfo_v1.sbatch)
embedding_job=$(tracked_sbatch --parsable --dependency="afterok:${prepare_job}" \
  scripts/slurm_sina_single_stock_cninfo_embeddings_v1.sbatch)
audit_job=$(tracked_sbatch --parsable --dependency="afterok:${embedding_job}" \
  scripts/slurm_audit_sina_single_stock_cninfo_v1.sbatch)

echo "prepare=${prepare_job} embedding=${embedding_job} audit=${audit_job} tasks=256 concurrency=16 device=cpu memory_per_task=32G"
