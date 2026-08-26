#!/bin/bash
set -euo pipefail
REPO=/mnt/lustre3/home/gaozh/llm_return
ROOT=${ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1}
DATASET=${DATASET:-sina}
SHARDS=${SHARDS:-64}
if [[ "${DATASET}" == "cninfo" ]]; then
  DEFAULT_PANEL=${REPO}/data/processed/cninfo_full_classification_panel.parquet
else
  DEFAULT_PANEL=/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/classification/sina_single_stock_classification_panel.parquet
fi
PANEL=${PANEL:-${DEFAULT_PANEL}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/soft_direction_tokens_${DATASET}_v1}
MANIFEST=${MANIFEST:-${OUTPUT_ROOT}/manifest.tsv}
MATRIX_MANIFEST=${MATRIX_MANIFEST:-${OUTPUT_ROOT}/matrix_manifest.tsv}
CONCURRENCY=${CONCURRENCY:-64}
cd "${REPO}"
mkdir -p logs/positive_token_soft_cluster "${OUTPUT_ROOT}"
.venv/bin/python scripts/build_positive_prompt_matrix_manifest.py \
  --output "${MATRIX_MANIFEST}" --dataset "${DATASET}" --shards "${SHARDS}" \
  --direction-only
.venv/bin/python scripts/build_positive_token_soft_cluster_manifest.py \
  --root "${ROOT}" --panel "${PANEL}" --output "${MANIFEST}" \
  --dataset "${DATASET}" --allow-missing
.venv/bin/python -m py_compile scripts/run_positive_token_soft_cluster_fold.py \
  scripts/build_positive_token_soft_cluster_manifest.py scripts/combine_positive_token_soft_cluster.py
bash -n scripts/slurm_positive_token_soft_cluster_fold.sbatch \
  scripts/slurm_positive_token_soft_cluster_evaluate.sbatch \
  scripts/slurm_positive_token_soft_cluster_summary.sbatch
matrix_dependency=()
if [[ -n "${EMBEDDING_DEPENDENCY:-}" ]]; then
  matrix_dependency+=(--dependency="afterok:${EMBEDDING_DEPENDENCY}")
fi
matrix_job=$(sbatch --parsable "${matrix_dependency[@]}" --array="0-15%16" \
  --export="ALL,MANIFEST=${MATRIX_MANIFEST},EMBEDDINGS_ROOT=${ROOT}/embeddings,OUTPUT_ROOT=${ROOT}/matrices" \
  scripts/slurm_build_positive_prompt_matrix.sbatch)
fold_job=$(sbatch --parsable --dependency="afterok:${matrix_job}" --array="0-143%${CONCURRENCY}" \
  --export="ALL,MANIFEST=${MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT}" \
  scripts/slurm_positive_token_soft_cluster_fold.sbatch)
eval_job=$(sbatch --parsable --dependency="afterok:${fold_job}" --array="0-15%8" \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT}" \
  scripts/slurm_positive_token_soft_cluster_evaluate.sbatch)
summary_job=$(sbatch --parsable --dependency="afterok:${eval_job}" \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT}" \
  scripts/slurm_positive_token_soft_cluster_summary.sbatch)
echo "matrices=${matrix_job} folds=${fold_job} evaluation=${eval_job} summary=${summary_job} output=${OUTPUT_ROOT}"
