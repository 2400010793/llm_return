#!/bin/bash
#SBATCH --job-name=qwen-cninfo-matrix
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/qwen_cninfo_strict/%A_%a.matrix.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/qwen_cninfo_strict/%A_%a.matrix.err

set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
: "${QWEN_ROOT:?export QWEN_ROOT}"
: "${INTERSECTION_ROOT:?export INTERSECTION_ROOT}"
: "${MATRIX_ROOT:?export MATRIX_ROOT}"

prompts=(future_return return loss excess_return profit)
kinds=(target_token article_mean)
prompt=${prompts[$((SLURM_ARRAY_TASK_ID / 2))]}
kind=${kinds[$((SLURM_ARRAY_TASK_ID % 2))]}
output="${MATRIX_ROOT}/masked_short_${prompt}_${kind}.npy"
mkdir -p "${MATRIX_ROOT}" logs/qwen_cninfo_strict

.venv/bin/python scripts/build_qwen_cninfo_strict_matrix.py \
  --root "${QWEN_ROOT}" \
  --manifest "${INTERSECTION_ROOT}/manifest.json" \
  --prompt "${prompt}" --variant masked_short --kind "${kind}" \
  --output "${output}"
