#!/bin/bash
#SBATCH --job-name=qwen-linear-final
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_qwen_linear/%j_final.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_qwen_linear/%j_final.err

set -euo pipefail
cd /data/alpha_team2/shares/llm_return
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
: "${STRICT_MANIFEST:?export STRICT_MANIFEST}"
: "${ROLLING_MANIFEST:?export ROLLING_MANIFEST}"
ROOT=reports/regression/qwen_2018_2026_linear_v1
.venv/bin/python scripts/summarize_qwen_linear_study.py \
  --selected "${ROOT}/selected_candidates.json" \
  --strict-manifest "${STRICT_MANIFEST}" --rolling-manifest "${ROLLING_MANIFEST}" \
  --panel data/processed/cninfo_full_o2o_residual_panel.parquet \
  --cluster-strict "${ROOT}/cluster_strict_6_2_1" \
  --cluster-rolling "${ROOT}/cluster_rolling_3_1_1" --output-root "${ROOT}"

for protocol in strict_2026 rolling_3_1_1; do
  predictions="${ROOT}/stock_day_predictions.parquet"
  [[ "${protocol}" == strict_2026 ]] && predictions="${ROOT}/strict_2026_stock_day_predictions.parquet"
  simple_dir="${ROOT}/simple_states/${protocol}"
  mkdir -p "${simple_dir}"
  .venv/bin/python scripts/evaluate_predictions_simple_states.py "${predictions}" \
    --output-dir "${simple_dir}" --factor-id "qwen_linear_${protocol}" \
    --label-lag 1 --direction 1 --top-n 100 --min-amount 20000000 \
    --max-missing-weight 0.05 || echo "simple_states failed for ${protocol}" >&2
  for cost in paper china_a_5bp; do
    cost_args=(--cost-model paper)
    [[ "${cost}" == china_a_5bp ]] && cost_args=(--cost-model china-a --commission-bps 5 --stamp-duty-bps 0 --slippage-bps 0)
    for side in long_short long short; do
      side_args=()
      [[ "${side}" == long ]] && side_args=(--long-only)
      [[ "${side}" == short ]] && side_args=(--short-only)
      .venv/bin/python scripts/run_portfolio_strategy.py "${predictions}" \
        --market-data data/processed/cninfo_full_o2o_market.parquet \
        --market-return-column tradable_open_to_open_return \
        --output-dir "${ROOT}/portfolio/${protocol}/${cost}/${side}" --gammas 1.0 \
        --can-buy-column can_buy --can-sell-column can_sell --eligible-column eligible_signal \
        --missing-return zero "${cost_args[@]}" "${side_args[@]}"
    done
  done
done
.venv/bin/python scripts/collect_qwen_backtests.py --root "${ROOT}"
