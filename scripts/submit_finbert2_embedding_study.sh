#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name finbert2-embedding-study \
    --purpose "Generate bounded FinBERT2-base token/pooled embeddings and run strict rolling classification and regression" \
    --input data/processed/cleaned/cninfo_prompt_bundle_existing_2018_2026_clean_v1/prompt_v3_fixed_parts \
    --input data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017_clean_v1/prompt_v3_fixed_parts \
    --input data/processed/cninfo_full_classification_panel_2010_2026.parquet \
    --input data/processed/cninfo_full_o2o_panel_2010_2026_hfq.parquet \
    --output data/processed/pooled_finbert2_embeddings_2010_2026_v1 \
    --output reports/classification/pooled_embeddings/finbert2 \
    --output reports/regression/pooled_embeddings/finbert2 \
    --related-file scripts/prefetch_finbert2_models.py \
    --related-file scripts/build_finbert2_evaluation_manifests.py \
    --related-file scripts/slurm_finbert2_prefetch.sbatch \
    --related-file scripts/slurm_finbert2_smoke.sbatch \
    --related-file scripts/slurm_finbert2_embeddings.sbatch \
    --related-file scripts/submit_finbert2_embedding_study.sh \
    --tag finbert2 --tag embedding --tag paper-replication --tag bounded-memory --seed 42 \
    -- bash "$0" "$@"
fi

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

embedding_concurrency=${EMBEDDING_CONCURRENCY:-8}
evaluation_concurrency=${EVALUATION_CONCURRENCY:-4}
mkdir -p logs/slurm_finbert2 reports/finbert2 \
  data/processed/pooled_finbert2_embeddings_2010_2026_v1
.venv/bin/python scripts/build_finbert2_evaluation_manifests.py

fetch_job=$(tracked_sbatch --parsable scripts/slurm_finbert2_prefetch.sbatch)
smoke_job=$(tracked_sbatch --parsable --dependency="afterok:${fetch_job}" \
  scripts/slurm_finbert2_smoke.sbatch)
plain_job=$(tracked_sbatch --parsable --dependency="afterok:${smoke_job}" \
  --array="0-63%${embedding_concurrency}" --export=ALL,VARIANT_GROUP=plain \
  scripts/slurm_finbert2_embeddings.sbatch)
prompt_job=$(tracked_sbatch --parsable --dependency="afterok:${smoke_job}" \
  --array="0-127%${embedding_concurrency}" --export=ALL,VARIANT_GROUP=prompted \
  scripts/slurm_finbert2_embeddings.sbatch)

paper_cls=$(tracked_sbatch --parsable --dependency="afterok:${plain_job}" \
  --array="0-0%1" \
  --export=ALL,MANIFEST=configs/generated/finbert2_paper_classification.tsv,PANEL=data/processed/cninfo_full_classification_panel_2010_2026.parquet,EXPECTED_ROWS=903665,SEARCH_STAGE=coarse \
  scripts/slurm_pooled_embedding_classification.sbatch)
paper_reg=$(tracked_sbatch --parsable --dependency="afterok:${plain_job}" \
  --array="0-0%1" \
  --export=ALL,MANIFEST=configs/generated/finbert2_paper_regression.tsv,PANEL=data/processed/cninfo_full_o2o_panel_2010_2026_hfq.parquet,EXPECTED_EMBEDDING_ROWS=903665,SEARCH_STAGE=coarse,SELECTION_CORRELATION=spearman \
  scripts/slurm_pooled_embedding_regression.sbatch)
pool_cls=$(tracked_sbatch --parsable --dependency="afterok:${prompt_job}" \
  --array="0-13%${evaluation_concurrency}" \
  --export=ALL,MANIFEST=configs/generated/finbert2_pooling_classification.tsv,PANEL=data/processed/cninfo_full_classification_panel_2010_2026.parquet,EXPECTED_ROWS=903665,SEARCH_STAGE=coarse \
  scripts/slurm_pooled_embedding_classification.sbatch)
pool_reg=$(tracked_sbatch --parsable --dependency="afterok:${prompt_job}" \
  --array="0-13%${evaluation_concurrency}" \
  --export=ALL,MANIFEST=configs/generated/finbert2_pooling_regression.tsv,PANEL=data/processed/cninfo_full_o2o_panel_2010_2026_hfq.parquet,EXPECTED_EMBEDDING_ROWS=903665,SEARCH_STAGE=coarse,SELECTION_CORRELATION=spearman \
  scripts/slurm_pooled_embedding_regression.sbatch)

echo "prefetch=${fetch_job} smoke=${smoke_job}"
echo "plain_embedding=${plain_job} prompted_embedding=${prompt_job} concurrency_each=${embedding_concurrency}"
echo "paper_classification=${paper_cls} paper_regression=${paper_reg}"
echo "pooling_classification=${pool_cls} pooling_regression=${pool_reg} evaluation_concurrency=${evaluation_concurrency}"
