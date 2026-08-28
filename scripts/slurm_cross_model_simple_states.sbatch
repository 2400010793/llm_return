#!/bin/bash
#SBATCH --job-name=xmodel-backtest
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/cross_model_span/%A_%a_bt.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/cross_model_span/%A_%a_bt.err
set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?array task required}"
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
SUMMARY_ROOT=${SUMMARY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/cross_model_span_v3/summary}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${SUMMARY_ROOT}/simple_states_manifest.tsv")
IFS=$'\t' read -r task_id predictions factor_id prediction_column <<< "${line%$'\r'}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || exit 2
cd "${REPO_ROOT}"
.venv/bin/python scripts/evaluate_predictions_simple_states.py "${predictions}" \
  --output-dir "${SUMMARY_ROOT}/simple_states/results/${factor_id}" \
  --factor-id "${factor_id}" --prediction-column "${prediction_column}" \
  --prediction-date-role execution --direction 1 --label-lag 1 --top-n 100 \
  --min-count 20 --min-amount 20000000 --max-missing-weight 0.05
