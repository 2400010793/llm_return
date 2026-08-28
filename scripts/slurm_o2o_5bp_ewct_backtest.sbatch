#!/bin/bash
#SBATCH --job-name=o2o-5bp-ewct
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_o2o_5bp_ewct/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_o2o_5bp_ewct/%A_%a.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?5bp backtest array must be submitted through task_tracker.py}"
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
market=${MARKET:-data/processed/cninfo_full_o2o_market_2010_2026_hfq.parquet}
output_root=${OUTPUT_ROOT:-reports/strategy/o2o_5bp_ewct_hfq}
base=${output_root}/${name}
mkdir -p "${base}" logs/slurm_o2o_5bp_ewct

# Symmetric 5bp buy/sell assumption: commission=5bp, no sell-only stamp duty.
gammas=${GAMMAS_ENCODED:-1.0:0.7:0.5:0.4:0.3:0.2:0.1}
gammas=${gammas//:/,}
for sleeve in paper_long_short china_a_long_only; do
  args=()
  [[ "${sleeve}" == "china_a_long_only" ]] && args+=(--long-only)
  /usr/bin/python3 scripts/run_portfolio_strategy.py "${predictions}" \
    --market-data "${market}" --market-return-column tradable_open_to_open_return \
    --output-dir "${base}/${sleeve}" --gammas "${gammas}" \
    --missing-return zero \
    --cost-model china-a --commission-bps 5 --stamp-duty-bps 0 --slippage-bps 0 \
    --can-buy-column can_buy --can-sell-column can_sell \
    --eligible-column eligible_signal "${args[@]}"
done
