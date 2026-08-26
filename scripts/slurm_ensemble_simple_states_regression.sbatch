#!/bin/bash
#SBATCH --job-name=ss-reg-ensemble
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_simple_states_seeds/%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_simple_states_seeds/%j.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?ensemble must use scripts/task_tracker.py}"
OUTPUT=${OUTPUT:-reports/regression/pooled_embeddings/simple_states_aligned_best_seeds_v1/seed_43_47_ensemble.stock_day_predictions.parquet}
mkdir -p logs/slurm_simple_states_seeds "$(dirname "${OUTPUT}")"
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

inputs=()
for seed in 43 44 45 46 47; do
  inputs+=(--input "reports/regression/pooled_embeddings/simple_states_aligned_best_seeds_v1/seed_${seed}.stock_day_predictions.parquet")
done
/usr/bin/python3 scripts/ensemble_regression_predictions.py \
  "${inputs[@]}" --output "${OUTPUT}"
