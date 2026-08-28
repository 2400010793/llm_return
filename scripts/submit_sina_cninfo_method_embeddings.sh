#!/bin/bash
set -euo pipefail
cd /home/team/llm_return

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name sina-cninfo-method-embeddings-v1 \
    --purpose "Generate CNINFO-style segment pools and every contextual Prompt token for Sina" \
    --input /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cleaned/sina_single_stock_clean.parquet \
    --input /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/classification/sina_single_stock_classification_panel.parquet \
    --output /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/inputs \
    --output /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/embeddings \
    --output reports/sina_cninfo_method_embedding_audit_v1.json \
    --related-file scripts/build_sina_cninfo_method_inputs.py \
    --related-file scripts/run_sina_cninfo_method_embeddings.py \
    --related-file scripts/audit_sina_cninfo_method_embeddings.py \
    --related-file scripts/slurm_sina_cninfo_method_embeddings.sbatch \
    --related-file scripts/build_sina_cninfo_method_downstream_manifests.py \
    --related-file scripts/slurm_launch_sina_cninfo_method_downstream.sbatch \
    --tag sina --tag prompt-token --tag cninfo-method --tag memory-safe --seed 42 \
    -- bash "$0" "$@"
fi

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

mkdir -p logs/slurm_sina_cninfo_method
.venv/bin/python scripts/build_sina_cninfo_method_inputs.py
embed_job=$(tracked_sbatch --parsable scripts/slurm_sina_cninfo_method_embeddings.sbatch)
audit_job=$(tracked_sbatch --parsable --dependency="afterok:${embed_job}" \
  scripts/slurm_audit_sina_cninfo_method_embeddings.sbatch)
launch_job=$(tracked_sbatch --parsable --dependency="afterok:${audit_job}" \
  scripts/slurm_launch_sina_cninfo_method_downstream.sbatch)
echo "embedding=${embed_job} audit=${audit_job} downstream_launcher=${launch_job} tasks=16 concurrency=4 memory_per_task=24G"
