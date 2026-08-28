#!/bin/bash
#SBATCH --job-name=o2o-final-bt
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_o2o_final/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_o2o_final/%A_%a.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?backtest array must be submitted through scripts/task_tracker.py}"
: "${MANIFEST:?export MANIFEST}"
: "${SLURM_ARRAY_TASK_ID:?run as an array}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id embedding_root model variant feature target regressor run_mode reducer components output <<< "${line}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || { echo "manifest task mismatch" >&2; exit 2; }

predictions=${output%.json}.stock_day_predictions.parquet
[[ -s "${predictions}" ]] || { echo "missing predictions: ${predictions}" >&2; exit 2; }
name=$(basename "${output%.json}")
market=data/processed/cninfo_full_o2o_market.parquet
base=reports/strategy/o2o_final_2026/${name}
mkdir -p "${base}" logs/slurm_o2o_final

# Paper-comparable one-day long-short strategy with symmetric 10 bps costs.
/usr/bin/python3 scripts/run_portfolio_strategy.py "${predictions}" \
  --market-data "${market}" --market-return-column tradable_open_to_open_return \
  --output-dir "${base}/paper_long_short" --gammas 1.0 \
  --can-buy-column can_buy --can-sell-column can_sell \
  --eligible-column eligible_signal --cost-model paper

# Implementable A-share companion: long-only, 3 bps commission and 5 bps sell stamp duty.
/usr/bin/python3 scripts/run_portfolio_strategy.py "${predictions}" \
  --market-data "${market}" --market-return-column tradable_open_to_open_return \
  --output-dir "${base}/china_a_long_only" --gammas 1.0 --long-only \
  --can-buy-column can_buy --can-sell-column can_sell \
  --eligible-column eligible_signal --cost-model china-a \
  --commission-bps 3 --stamp-duty-bps 5
