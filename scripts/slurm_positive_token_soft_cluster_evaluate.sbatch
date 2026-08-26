#!/bin/bash
#SBATCH --job-name=positive-token-eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/positive_token_soft_cluster/%A_%a_eval.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/positive_token_soft_cluster/%A_%a_eval.err

set -euo pipefail
: "${OUTPUT_ROOT:?OUTPUT_ROOT required}"
: "${SLURM_ARRAY_TASK_ID:?array task required}"
REPO=/mnt/lustre3/home/gaozh/llm_return
index=${SLURM_ARRAY_TASK_ID}
representation=
if [[ -n "${CONFIG_MANIFEST:-}" ]]; then
  line=$(sed -n "$((index + 2))p" "${CONFIG_MANIFEST}")
  IFS=$'\t' read -r task_id prompt model variant representation projection_mode <<< "${line%$'\r'}"
  [[ "${task_id}" == "${index}" ]] || { echo "config manifest task mismatch" >&2; exit 2; }
else
  prompts=(profit excess_return return loss)
  models=(roberta bge_m3)
  variants=(short masked_short)
  variant=${variants[$((index % 2))]}
  model=${models[$(((index / 2) % 2))]}
  prompt=${prompts[$((index / 4))]}
fi
projection_mode=${projection_mode:-pca}
fold_root=${OUTPUT_ROOT}/folds/${prompt}/${model}/${variant}
combined=${OUTPUT_ROOT}/combined/${prompt}/${model}/${variant}/stock_day_predictions.parquet
factor_suffix=${prompt}_${model}_${variant}
if [[ -n "${representation}" ]]; then
  fold_root=${fold_root}/${representation}
  combined=${OUTPUT_ROOT}/combined/${prompt}/${model}/${variant}/${representation}/stock_day_predictions.parquet
  factor_suffix=${factor_suffix}_${representation}
fi
if [[ "${projection_mode}" == raw ]]; then
  fold_root=${fold_root}/raw
  combined=${combined%/stock_day_predictions.parquet}/raw/stock_day_predictions.parquet
  factor_suffix=${factor_suffix}_raw
fi
cd "${REPO}"
.venv/bin/python scripts/combine_positive_token_soft_cluster.py --fold-root "${fold_root}" --output "${combined}"

market=data/processed/cninfo_full_o2o_market_2010_2026_hfq.parquet
gammas=${GAMMAS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
for objective in rank_ic long_short long_only; do
  column=prediction_${objective}
  factor_id=soft_${factor_suffix}_${objective}
  simple_dir=${OUTPUT_ROOT}/simple_states/${factor_id}
  .venv/bin/python scripts/evaluate_predictions_simple_states.py "${combined}" \
    --output-dir "${simple_dir}" --factor-id "${factor_id}" --prediction-column "${column}" \
    --prediction-date-role execution --direction 1 --label-lag 1 --top-n 100 \
    --min-count 20 --min-amount 20000000 --max-missing-weight 0.05
  for cost in paper_10bp china_a_5bp; do
    cost_args=(--cost-model paper)
    [[ "${cost}" == china_a_5bp ]] && cost_args=(--cost-model china-a --commission-bps 5 --stamp-duty-bps 0 --slippage-bps 0)
    for mode in long_only short_only long_short; do
      side=()
      [[ "${mode}" == long_only ]] && side+=(--long-only)
      [[ "${mode}" == short_only ]] && side+=(--short-only)
      .venv/bin/python scripts/run_portfolio_strategy.py "${combined}" \
        --market-data "${market}" --market-return-column tradable_open_to_open_return \
        --output-dir "${OUTPUT_ROOT}/portfolio/${factor_id}/${cost}/${mode}" \
        --prediction-column "${column}" --gammas "${gammas}" --missing-return zero \
        "${cost_args[@]}" --can-buy-column can_buy --can-sell-column can_sell \
        --eligible-column eligible_signal "${side[@]}"
    done
  done
done
