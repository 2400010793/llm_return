#!/bin/bash
#SBATCH --job-name=emb-hdb-full
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/correct_embedding_hdbscan/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/correct_embedding_hdbscan/%A_%a.err
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return
mkdir -p logs/correct_embedding_hdbscan
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
MANIFEST=${CORRECT_EMB_MANIFEST:?set CORRECT_EMB_MANIFEST}
OFFSET=${ARRAY_OFFSET:-0}
IFS=$'\t' read -r source dataset model prompt variant feature test_year mode panel matrix metadata embedding_root pooled_feature <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + OFFSET + 2))p" "$MANIFEST")"
ARGS=(--panel "$panel" --output-root /mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/results_all_embedding_cluster_regression --dataset "$dataset" --model "$model" --prompt "$prompt" --variant "$variant" --feature "$feature" --test-year "$test_year" --mode "$mode" --pca-sample-rows 20000 --cluster-sample-rows 20000)
if [[ "$source" == matrix ]]; then
  ARGS+=(--matrix "$matrix" --metadata "$metadata")
else
  ARGS+=(--embedding-root "$embedding_root" --pooled-feature "$pooled_feature" --expected-rows 903665 --max-matrix-gib 16)
fi
.venv/bin/python scripts/run_all_embedding_hdbscan_fold.py "${ARGS[@]}"
