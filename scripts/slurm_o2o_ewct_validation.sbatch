#!/bin/bash
#SBATCH --job-name=o2o-ewct-val
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_o2o_ewct/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_o2o_ewct/%A_%a.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?validation array must be tracked}"
: "${MANIFEST:?export MANIFEST}"
: "${SLURM_ARRAY_TASK_ID:?run as an array}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id model_id validation_predictions confirmation_predictions quantiles output_dir <<< "${line}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || { echo "manifest task mismatch" >&2; exit 2; }

/usr/bin/python3 scripts/run_portfolio_strategy.py "${validation_predictions}" \
  --market-data data/processed/cninfo_full_o2o_market.parquet \
  --market-return-column open_to_open_return --output-dir "${output_dir}" \
  --quantiles "${quantiles}" --gammas 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --min-stocks-per-day "${quantiles}" \
  --can-buy-column can_buy --can-sell-column can_sell \
  --eligible-column eligible_signal --missing-return zero --cost-model paper
