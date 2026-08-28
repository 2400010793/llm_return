#!/bin/bash
#SBATCH --job-name=tb-mlp-ensemble
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_title_body_ensemble/%j.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_title_body_ensemble/%j.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?ensemble job must use scripts/task_tracker.py}"
: "${INPUT_ROOT:?export INPUT_ROOT}"
: "${OUTPUT:?export OUTPUT}"
mkdir -p logs/slurm_title_body_ensemble "$(dirname "${OUTPUT}")"
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

/usr/bin/python3 scripts/ensemble_title_body_mlp_predictions.py \
  --prediction "${INPUT_ROOT}/roberta_short_title_body_mean_simple_mlp_select_accuracy.predictions.parquet" --seed 42 \
  --prediction "${INPUT_ROOT}/seed_stability/roberta_short_title_body_mean_simple_mlp_select_accuracy_seed43.predictions.parquet" --seed 43 \
  --prediction "${INPUT_ROOT}/seed_stability/roberta_short_title_body_mean_simple_mlp_select_accuracy_seed44.predictions.parquet" --seed 44 \
  --prediction "${INPUT_ROOT}/seed_stability/roberta_short_title_body_mean_simple_mlp_select_accuracy_seed45.predictions.parquet" --seed 45 \
  --prediction "${INPUT_ROOT}/seed_stability/roberta_short_title_body_mean_simple_mlp_select_accuracy_seed46.predictions.parquet" --seed 46 \
  --output "${OUTPUT}"
