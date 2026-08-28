#!/bin/bash
set -euo pipefail
cd /home/team/llm_return

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name sina-pooling-ablation-v1 \
    --purpose "Compare mean, CLS/BOS, and max pooling on aligned Sina news, including XLM-R" \
    --input /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/prompts/sina_single_stock_prompt_variants.parquet \
    --input /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/classification/sina_single_stock_classification_panel.parquet \
    --output /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/embeddings/single_stock/pooling_ablation_v1 \
    --output /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/classification/results_pooling_ablation_v1 \
    --output /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/regression/results_pooling_ablation_v1 \
    --related-file scripts/run_sina_pooling_embeddings.py \
    --related-file scripts/slurm_sina_pooling_embeddings.sbatch \
    --related-file scripts/slurm_sina_pooling_classification.sbatch \
    --related-file scripts/slurm_sina_pooling_regression.sbatch \
    --tag sina --tag pooling --tag paper-replication --seed 42 \
    -- bash "$0" "$@"
fi

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

mkdir -p logs/slurm_sina_pooling
embed_job=$(tracked_sbatch --parsable scripts/slurm_sina_pooling_embeddings.sbatch)
cls_job=$(tracked_sbatch --parsable --dependency="afterok:${embed_job}" \
  scripts/slurm_sina_pooling_classification.sbatch)
reg_job=$(tracked_sbatch --parsable --dependency="afterok:${embed_job}" \
  scripts/slurm_sina_pooling_regression.sbatch)
echo "embedding=${embed_job} classification=${cls_job} regression=${reg_job}"
