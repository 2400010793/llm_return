#!/bin/bash
#SBATCH --job-name=umap-hdbscan-ret
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-8%3
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/umap_hdbscan/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/umap_hdbscan/%A_%a.err
set -euo pipefail
cd /data/alpha_team2/shares/llm_return
mkdir -p logs/umap_hdbscan
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
.venv/bin/python scripts/run_umap_hdbscan_regression.py \
  --panel /data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/classification/sina_single_stock_classification_panel.parquet \
  --matrix /data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/matrices/sina/bge_m3/loss/masked_short/loss_span/matrix.npy \
  --metadata /data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/matrices/sina/bge_m3/loss/masked_short/loss_span/metadata.parquet \
  --output-root /data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/results_umap_hdbscan/bge_m3_masked_short_loss_span \
  --test-year $((2018 + SLURM_ARRAY_TASK_ID)) \
  --target next_day_return
