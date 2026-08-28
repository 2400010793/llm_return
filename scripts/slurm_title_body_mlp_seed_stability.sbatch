#!/bin/bash
#SBATCH --job-name=tb-mlp-seeds
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%8
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_title_body_seeds/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_title_body_seeds/%A_%a.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?seed-stability array must be submitted through scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"
: "${PANEL:?export PANEL}"
: "${EMBEDDING_ROOT:?export EMBEDDING_ROOT}"
: "${OUTPUT_ROOT:?export OUTPUT_ROOT}"

mkdir -p logs/slurm_title_body_seeds "${OUTPUT_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

SEEDS=(43 44 45 46)
seed_index=$((SLURM_ARRAY_TASK_ID % 4))
SEED=${SEEDS[$seed_index]}
if (( SLURM_ARRAY_TASK_ID < 4 )); then
  SELECTION_METRIC=accuracy
else
  SELECTION_METRIC=balanced_accuracy
fi

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
  --expected-embedding-rows 903665 \
  --output "${OUTPUT_ROOT}/roberta_short_title_body_mean_simple_mlp_select_${SELECTION_METRIC}_seed${SEED}.json"
