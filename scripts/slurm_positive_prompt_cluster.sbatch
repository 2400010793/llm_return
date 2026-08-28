#!/bin/bash
#SBATCH --job-name=positive-cluster
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/positive_prompt_cluster/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/positive_prompt_cluster/%A_%a.err
set -euo pipefail
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
IFS=$'\t' read -r dataset prompt model variant representation shards <<< "${line%$'\r'}"
ROOT=/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_positive_v1
PANEL=$ROOT/../classification/sina_single_stock_classification_panel.parquet
MATRIX=$ROOT/matrices/$dataset/$model/$prompt/$variant/$representation
OUT=$ROOT/results_cluster/$dataset/${prompt}_${model}_${variant}_${representation}/stock_day_predictions.parquet
cd /data/alpha_team2/shares/llm_return; mkdir -p "$(dirname "$OUT")"
.venv/bin/python scripts/run_positive_prompt_cluster_regression.py "$PANEL" --matrix "$MATRIX/matrix.npy" --metadata "$MATRIX/metadata.parquet" --output "$OUT"
