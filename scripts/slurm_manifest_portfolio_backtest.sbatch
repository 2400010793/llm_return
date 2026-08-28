#!/bin/bash
#SBATCH --job-name=manifest-portfolio
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_manifest_portfolio/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_manifest_portfolio/%A_%a.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?portfolio array must use scripts/task_tracker.py}"
: "${MANIFEST:?export MANIFEST}"
: "${OUTPUT_ROOT:?export OUTPUT_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?run as an array job}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id config predictions <<< "${line}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || {
  echo "manifest task mismatch" >&2
  exit 2
}
[[ -s "${predictions}" ]] || {
  echo "missing predictions: ${predictions}" >&2
  exit 2
}

market=${MARKET:-data/processed/cninfo_full_o2o_market_2010_2026_hfq.parquet}
gammas=${GAMMAS_ENCODED:-0.1}
gammas=${gammas//:/,}
mkdir -p "${OUTPUT_ROOT}/${config}" logs/slurm_manifest_portfolio

for sleeve in long_only long_short; do
  side_args=()
  [[ "${sleeve}" == "long_only" ]] && side_args+=(--long-only)
  /usr/bin/python3 scripts/run_portfolio_strategy.py "${predictions}" \
    --market-data "${market}" \
    --market-return-column tradable_open_to_open_return \
    --output-dir "${OUTPUT_ROOT}/${config}/${sleeve}" \
    --gammas "${gammas}" \
    --missing-return zero \
    --cost-model china-a --commission-bps 5 \
    --stamp-duty-bps 0 --slippage-bps 0 \
    --can-buy-column can_buy --can-sell-column can_sell \
    --eligible-column eligible_signal "${side_args[@]}"
done
