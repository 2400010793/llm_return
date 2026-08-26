#!/bin/bash
#SBATCH --job-name=mlp-select
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%3
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_mlp_selection/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_mlp_selection/%A_%a.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?selection-metric array must be submitted through scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"
: "${PANEL:?export PANEL}"
: "${MATRIX:?export MATRIX}"
: "${OUTPUT_ROOT:?export OUTPUT_ROOT}"

mkdir -p logs/slurm_mlp_selection "${OUTPUT_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) SELECTION_METRIC=accuracy ;;
  1) SELECTION_METRIC=balanced_accuracy ;;
  2) SELECTION_METRIC=auc ;;
  *) echo "unsupported array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

EXTRA_ARGS=()
if [[ -n "${MAX_ITER:-}" ]]; then EXTRA_ARGS+=(--max-iter "${MAX_ITER}"); fi

/usr/bin/python3 scripts/run_reduced_prompt_classification.py \
  "${PANEL}" \
  --matrix "${MATRIX}" \
  --model roberta \
  --variant masked_short \
  --classifier simple_mlp \
  --components 128 \
  --learning-rate-schedule constant \
  --early-stopping-metric "${SELECTION_METRIC}" \
  --final-epoch-multiplier 1.0 \
  --seed 42 \
  "${EXTRA_ARGS[@]}" \
  --output "${OUTPUT_ROOT}/roberta_masked_short_prompt_pca128_body_simple_mlp_select_${SELECTION_METRIC}.json"
