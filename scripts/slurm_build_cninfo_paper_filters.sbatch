#!/bin/bash
#SBATCH --job-name=cninfo-paper-filter
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_paper_filters/%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_paper_filters/%j.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?filter build must be tracked}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}
mkdir -p logs/slurm_paper_filters data/processed/paper_filters

/usr/bin/python3 scripts/build_cninfo_paper_filters.py \
  --panel data/processed/cninfo_full_o2o_residual_panel.parquet \
  --market-data data/processed/cninfo_full_o2o_market.parquet \
  --embedding-root data/processed/pooled_embeddings_v3 \
  --output-dir data/processed/paper_filters \
  --model roberta --variant masked_short --feature body_mean \
  --min-chars 100 --max-chars 100000 --lookback-trading-days 5 \
  --thresholds 0.8,0.9,0.95,0.98
