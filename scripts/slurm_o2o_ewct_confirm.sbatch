#!/bin/bash
#SBATCH --job-name=o2o-ewct-confirm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_o2o_ewct/%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_o2o_ewct/%j.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?confirmation job must be tracked}"
: "${SELECTION_OUTPUT:?export SELECTION_OUTPUT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

/usr/bin/python3 scripts/run_selected_o2o_ewct.py \
  --selection "${SELECTION_OUTPUT}" \
  --market-data data/processed/cninfo_full_o2o_market.parquet \
  --output-root reports/strategy/o2o_ewct_2026_confirmation \
  --confirmation-report reports/strategy/o2o_ewct_2026_confirmation.json
