#!/bin/bash
# Build cross-sectional O2O targets from the full eligible 816-stock market panel.
#SBATCH --job-name=o2o-resid-data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_o2o_residual/%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_o2o_residual/%j.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"

SOURCE_PANEL=${SOURCE_PANEL:-data/processed/cninfo_full_o2o_panel.parquet}
MARKET_DATA=${MARKET_DATA:-data/processed/cninfo_full_o2o_market.parquet}
OUTPUT_PANEL=${OUTPUT_PANEL:-data/processed/cninfo_full_o2o_residual_panel.parquet}
mkdir -p logs/slurm_o2o_residual "$(dirname "${OUTPUT_PANEL}")"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

/usr/bin/python3 scripts/build_o2o_residual_panel.py \
  "${SOURCE_PANEL}" --market-data "${MARKET_DATA}" --output "${OUTPUT_PANEL}" \
  --winsor-lower 0.01 --winsor-upper 0.99 --min-stocks-per-day 5
