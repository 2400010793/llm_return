#!/bin/bash
#SBATCH --job-name=tb-mlp-year
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-59%60
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_title_body_year_parallel/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_title_body_year_parallel/%A_%a.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?year-parallel array must be submitted through scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"
: "${PANEL:?export PANEL}"
: "${EMBEDDING_ROOT:?export EMBEDDING_ROOT}"
: "${OUTPUT_ROOT:?export OUTPUT_ROOT}"

mkdir -p logs/slurm_title_body_year_parallel "${OUTPUT_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

SEEDS=(42 43 44 45 46)
YEARS=(2021 2022 2023 2024 2025 2026)
year_index=$((SLURM_ARRAY_TASK_ID % 6))
configuration_index=$((SLURM_ARRAY_TASK_ID / 6))
seed_index=$((configuration_index % 5))
metric_index=$((configuration_index / 5))
SEED=${SEEDS[$seed_index]}
TEST_YEAR=${YEARS[$year_index]}
if (( metric_index == 0 )); then
  SELECTION_METRIC=accuracy
else
  SELECTION_METRIC=balanced_accuracy
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${SELECTION_METRIC}/seed${SEED}"
mkdir -p "${OUTPUT_DIR}"

/usr/bin/python3 scripts/run_pooled_embedding_classification.py \
  "${PANEL}" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --model roberta \
  --variant short \
  --feature title_body_mean \
  --classifier simple_mlp \
  --reducer none \
  --reducer-components 128 \
  --scaler auto \
  --train-target-column next_day_return \
  --evaluation-target-column next_day_return \
  --date-column entry_date \
  --panel-row-index-column row_index \
  --search-stage fine \
  --learning-rate-schedule constant \
  --early-stopping-metric "${SELECTION_METRIC}" \
  --final-epoch-multiplier 1.0 \
  --seed "${SEED}" \
  --test-years "${TEST_YEAR}" \
  --expected-embedding-rows 903665 \
  --output "${OUTPUT_DIR}/roberta_short_title_body_mean_simple_mlp_${SELECTION_METRIC}_seed${SEED}_${TEST_YEAR}.json"
