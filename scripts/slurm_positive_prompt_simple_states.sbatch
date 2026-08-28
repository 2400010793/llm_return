#!/bin/bash
#SBATCH --job-name=positive-backtest
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/positive_prompt_backtest/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/positive_prompt_backtest/%A_%a.err
set -euo pipefail
: "${MANIFEST:?MANIFEST required}"
TOP_N=${TOP_N:-100}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
IFS=$'\t' read -r task_id predictions factor_id prediction_column <<< "${line%$'\r'}"
cd /data/alpha_team2/shares/llm_return
mkdir -p "$(dirname "$predictions")/../simple_states/results/$factor_id"
.venv/bin/python scripts/evaluate_predictions_simple_states.py "$predictions" \
  --output-dir "$(dirname "$MANIFEST")/simple_states/results/$factor_id" \
  --factor-id "$factor_id" --prediction-column "$prediction_column" \
  --prediction-date-role execution --direction 1 --label-lag 1 --top-n "${TOP_N}" \
  --min-count 20 --min-amount 20000000 --max-missing-weight 0.05
