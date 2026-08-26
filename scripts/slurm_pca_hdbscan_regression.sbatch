#!/bin/bash
#SBATCH --job-name=pca-hdbscan-ret
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-8%3
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/pca_hdbscan/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/pca_hdbscan/%A_%a.err
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return
mkdir -p logs/pca_hdbscan
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
.venv/bin/python scripts/run_umap_hdbscan_regression.py \
  --panel /mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/classification/sina_single_stock_classification_panel.parquet \
  --matrix /mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/matrices/sina/bge_m3/loss/masked_short/loss_span/matrix.npy \
  --metadata /mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/matrices/sina/bge_m3/loss/masked_short/loss_span/metadata.parquet \
  --output-root /mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/results_pca_hdbscan_fixed_v1/bge_m3_masked_short_loss_span \
  --test-year $((2018 + SLURM_ARRAY_TASK_ID)) \
  --target next_day_return --mode pca --fixed-min-cluster-size 50 --fixed-min-samples 10 --fixed-alpha 1
