#!/bin/bash
#SBATCH --job-name=lr-anneal
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_lr_anneal/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_lr_anneal/%A_%a.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?annealing array must be submitted through scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"
: "${PANEL:?export PANEL}"
: "${MATRIX:?export MATRIX}"
: "${OUTPUT_ROOT:?export OUTPUT_ROOT}"

mkdir -p logs/slurm_lr_anneal "${OUTPUT_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) SCHEDULE=constant ;;
  1) SCHEDULE=cosine ;;
  2) SCHEDULE=inverse_time ;;
  *) echo "unsupported array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

EXTRA_ARGS=()
if [[ -n "${MAX_FOLDS:-}" ]]; then EXTRA_ARGS+=(--max-folds "${MAX_FOLDS}"); fi
if [[ -n "${MAX_ITER:-}" ]]; then EXTRA_ARGS+=(--max-iter "${MAX_ITER}"); fi
if [[ -n "${FINAL_EPOCH_MULTIPLIER:-}" ]]; then EXTRA_ARGS+=(--final-epoch-multiplier "${FINAL_EPOCH_MULTIPLIER}"); fi
if [[ -n "${EARLY_STOPPING_METRIC:-}" ]]; then EXTRA_ARGS+=(--early-stopping-metric "${EARLY_STOPPING_METRIC}"); fi

/usr/bin/python3 scripts/run_reduced_prompt_classification.py \
  "${PANEL}" \
  --matrix "${MATRIX}" \
  --model roberta \
  --variant masked_short \
  --classifier simple_mlp \
  --components 128 \
  --learning-rate-schedule "${SCHEDULE}" \
  --seed 42 \
  "${EXTRA_ARGS[@]}" \
  --output "${OUTPUT_ROOT}/roberta_masked_short_prompt_pca128_body_simple_mlp_${SCHEDULE}.json"
