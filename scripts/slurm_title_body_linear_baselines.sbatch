#!/bin/bash
#SBATCH --job-name=tb-linear-strict
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_title_body_linear/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_title_body_linear/%A_%a.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?linear baseline array must use scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?run as an array job}"
: "${PANEL:?export PANEL}"
: "${EMBEDDING_ROOT:?export EMBEDDING_ROOT}"
: "${OUTPUT_ROOT:?export OUTPUT_ROOT}"

mkdir -p logs/slurm_title_body_linear "${OUTPUT_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) CLASSIFIER=logistic ;;
  1) CLASSIFIER=linear_svm ;;
  *) echo "unsupported array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

/usr/bin/python3 scripts/run_pooled_embedding_classification.py \
  "${PANEL}" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --model roberta --variant short --feature title_body_mean \
  --classifier "${CLASSIFIER}" --reducer none --reducer-components 128 \
  --scaler auto \
  --train-target-column next_day_return \
  --evaluation-target-column next_day_return \
  --date-column entry_date --panel-row-index-column row_index \
  --search-stage fine --early-stopping-metric accuracy \
  --seed 42 --expected-embedding-rows 903665 \
  --output "${OUTPUT_ROOT}/roberta_short_title_body_mean_${CLASSIFIER}_accuracy_fine.json"
