#!/bin/bash
#SBATCH --job-name=qwen-cninfo-reg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/qwen_cninfo_strict/%A_%a.regression.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/qwen_cninfo_strict/%A_%a.regression.err

set -euo pipefail
cd /data/alpha_team2/shares/llm_return
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
: "${INTERSECTION_ROOT:?export INTERSECTION_ROOT}"
: "${MATRIX_ROOT:?export MATRIX_ROOT}"
: "${REGRESSION_ROOT:?export REGRESSION_ROOT}"

prompts=(future_return return loss excess_return profit)
kinds=(target_token article_mean)
prompt=${prompts[$((SLURM_ARRAY_TASK_ID / 2))]}
kind=${kinds[$((SLURM_ARRAY_TASK_ID % 2))]}
matrix="${MATRIX_ROOT}/masked_short_${prompt}_${kind}.npy"
output="${REGRESSION_ROOT}/masked_short_${prompt}_${kind}.json"
mkdir -p "${REGRESSION_ROOT}" logs/qwen_cninfo_strict
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

.venv/bin/python scripts/run_sina_precomputed_regression.py \
  "${INTERSECTION_ROOT}/intersection_panel.parquet" \
  --matrix "${matrix}" \
  --metadata "${INTERSECTION_ROOT}/intersection_metadata.parquet" \
  --embedding-model qwen3_embedding_8b --variant masked_short \
  --target-column next_day_open_to_open_return \
  --panel-row-index-column row_index \
  --metadata-row-index-column qwen_row_index --metadata-row-index-offset 1 \
  --reducer pca --reducer-components 128 \
  --regressor ridge --run-mode final-test --search-stage fine \
  --expected-rows 250344 --seed 42 --output "${output}"
