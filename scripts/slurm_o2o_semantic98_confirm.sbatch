#!/bin/bash
#SBATCH --job-name=o2o-sem98-confirm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_paper_filters/%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_paper_filters/%j.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?confirmation job must be tracked}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

predictions=${PREDICTIONS:-reports/regression/pooled_embeddings/o2o_paper_filters/semantic_098_winsor_huber_pca256_final_2026.stock_day_predictions.parquet}
output_dir=${OUTPUT_DIR:-reports/strategy/o2o_paper_filters_confirmation/semantic98_q5_gamma0p1}
quantiles=${QUANTILES:-5}
gamma=${GAMMA:-0.1}
[[ -s "${predictions}" ]] || { echo "missing predictions: ${predictions}" >&2; exit 2; }
/usr/bin/python3 scripts/run_portfolio_strategy.py "${predictions}" \
  --market-data data/processed/cninfo_full_o2o_market.parquet \
  --market-return-column open_to_open_return \
  --output-dir "${output_dir}" \
  --quantiles "${quantiles}" --min-stocks-per-day "${quantiles}" --gammas "${gamma}" \
  --can-buy-column can_buy --can-sell-column can_sell \
  --eligible-column eligible_signal --missing-return zero --cost-model paper
