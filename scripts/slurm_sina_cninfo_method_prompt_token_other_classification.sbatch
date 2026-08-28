#!/bin/bash
# Additional linear classifiers for the Prompt-token contextual embedding.
#SBATCH --job-name=sina-cninfo-token-other
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/home/team/llm_return/logs/slurm_sina_cninfo_method/%A_%a_token_other.out
#SBATCH --error=/home/team/llm_return/logs/slurm_sina_cninfo_method/%A_%a_token_other.err

set -euo pipefail
cd /home/team/llm_return
: "${TASK_RECORD_ID:?model array must be submitted through scripts/task_tracker.py}"
: "${MANIFEST:?export MANIFEST}"
: "${SLURM_ARRAY_TASK_ID:?run as an array}"
PANEL=${PANEL:-/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/classification/sina_single_stock_classification_panel.parquet}

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
mkdir -p "$(dirname "${output}")" logs/slurm_sina_cninfo_method
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

.venv/bin/python scripts/run_pooled_embedding_classification.py \
  "${PANEL}" \
  --embedding-root "${embedding_root}" \
  --model "${model}" \
  --variant "${variant}" \
  --feature "${feature}" \
  --classifier "${classifier}" \
  --reducer "${reducer}" \
  --reducer-components "${components:-128}" \
  --scaler auto \
  --train-target-column "${train_target}" \
  --evaluation-target-column "${evaluation_target}" \
  --date-column entry_date \
  --panel-row-index-column row_index \
  --expected-embedding-rows 4928 \
  --max-embedding-matrix-gib 12 \
  --search-stage coarse \
  --run-mode final-test \
  --seed 42 \
  --output "${output}"
