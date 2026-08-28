#!/bin/bash
# Submit the active paper-faithful China (HK) encoder, then its classification
# and O2O regression matrices. CKIP-BERT was retired after its 2026 OOS audit.
set -euo pipefail
cd /home/team/llm_return

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name paper-hk-priority \
    --purpose "Run the active XLM-R China-HK encoder and prioritized next-day/O2O predictors" \
    --input data/processed/cleaned/prompt_v3_fixed_parts \
    --input data/processed/cninfo_full_classification_panel.parquet \
    --output data/processed/pooled_paper_hk_embeddings_v1 \
    --output reports/classification/pooled_embeddings/paper_hk \
    --output reports/regression/pooled_embeddings/stock_day \
    --related-file scripts/prefetch_paper_hk_models.py \
    --related-file scripts/slurm_prefetch_paper_hk_models.sbatch \
    --related-file scripts/slurm_paper_hk_embeddings.sbatch \
    --related-file scripts/slurm_launch_completed_embedding_classification.sbatch \
    --related-file scripts/slurm_launch_completed_embedding_regression.sbatch \
    --tag paper-replication --tag china-hk --seed 42 \
    -- bash "$0" "$@"
fi

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

mkdir -p logs/slurm_paper_hk data/processed/pooled_paper_hk_embeddings_v1
.venv/bin/python -m py_compile \
  scripts/prefetch_paper_hk_models.py \
  scripts/run_short_pooled_embeddings.py \
  scripts/build_pooled_classification_manifest.py \
  scripts/build_pooled_regression_manifest.py
bash -n \
  scripts/slurm_prefetch_paper_hk_models.sbatch \
  scripts/slurm_paper_hk_embeddings.sbatch \
  scripts/slurm_launch_completed_embedding_classification.sbatch \
  scripts/slurm_launch_completed_embedding_regression.sbatch

fetch_job=$(tracked_sbatch --parsable scripts/slurm_prefetch_paper_hk_models.sbatch)
for model in xlm_roberta_large; do
  embed_job=$(tracked_sbatch --parsable \
    --dependency="afterok:${fetch_job}" \
    --array="0-31%4" \
    --export="ALL,MODEL=${model}" \
    scripts/slurm_paper_hk_embeddings.sbatch)
  cls_job=$(tracked_sbatch --parsable \
    --dependency="afterok:${embed_job}" \
    --export="ALL,MODEL=${model},VARIANT_GROUP=paper_hk,PHASES=paper_hk,CLASSIFICATION_CONCURRENCY=4" \
    scripts/slurm_launch_completed_embedding_classification.sbatch)
  reg_job=$(tracked_sbatch --parsable \
    --dependency="afterok:${embed_job}" \
    --export="ALL,MODEL=${model},VARIANT_GROUP=paper_hk,FEATURE=full_mean,TARGETS=next_day_open_to_open_return,REGRESSORS=ridge,huber_sgd,small_mlp,REDUCERS=none,pca:128,REGRESSION_CONCURRENCY=4" \
    scripts/slurm_launch_completed_embedding_regression.sbatch)
  echo "model=${model} embedding=${embed_job} classification_launcher=${cls_job} regression_launcher=${reg_job}"
done
echo "prefetch=${fetch_job}"
