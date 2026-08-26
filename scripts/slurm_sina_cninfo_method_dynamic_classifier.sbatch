#!/bin/bash
# Existing prompt-token embeddings -> dynamic gate -> rolling classifier.
# This worker never generates embeddings.
#SBATCH --job-name=sina-dyn-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-191%16
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_sina_dynamic_classifier/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_sina_dynamic_classifier/%A_%a.err

set -euo pipefail
cd /home/gaozh/llm_return
: "${TASK_RECORD_ID:?dynamic classifier array must be submitted through scripts/task_tracker.py}"
: "${MANIFEST:?export MANIFEST}"
: "${SLURM_ARRAY_TASK_ID:?run as an array}"
PANEL=${PANEL:-/home/gaozh/news_content_quality_20260812/classification/sina_single_stock_classification_panel.parquet}

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
if [[ -z "${line}" ]]; then
  echo "No task ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
  exit 2
fi
IFS=$'\t' read -r task_id embedding_root model variant feature classifier reducer components output train_target evaluation_target <<< "${line}"
if [[ "${task_id}" != "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "Manifest task mismatch: expected ${SLURM_ARRAY_TASK_ID}, found ${task_id}" >&2
  exit 2
fi
: "${train_target:?manifest must provide train_target}"
: "${evaluation_target:?manifest must provide evaluation_target}"
mkdir -p "$(dirname "${output}")" logs/slurm_sina_dynamic_classifier
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

.venv/bin/python scripts/run_pooled_embedding_classification.py \
  "${PANEL}" \
  --embedding-root "${embedding_root}" \
  --model "${model}" \
  --variant "${variant}" \
  --feature "${feature}" \
  --classifier "${classifier}" \
  --reducer "${reducer}" \
  --scaler auto \
  --train-target-column "${train_target}" \
  --evaluation-target-column "${evaluation_target}" \
  --date-column entry_date \
  --panel-row-index-column row_index \
  --expected-embedding-rows 4928 \
  --max-embedding-matrix-gib 12 \
  --dynamic-gate-device cpu \
  --dynamic-gate-epochs "${DYNAMIC_GATE_EPOCHS:-10}" \
  --dynamic-gate-batch-size "${DYNAMIC_GATE_BATCH_SIZE:-256}" \
  --search-stage coarse \
  --run-mode final-test \
  --seed "${SEED:-42}" \
  --output "${output}"
