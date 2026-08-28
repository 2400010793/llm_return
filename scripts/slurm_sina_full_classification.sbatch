#!/bin/bash
#SBATCH --job-name=sina-full-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-2%3
#SBATCH --output=/home/team/llm_return/logs/slurm_sina_full_classification/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_sina_full_classification/%A_%a.err

set -euo pipefail
cd /home/team/llm_return
: "${TASK_RECORD_ID:?classification must be submitted through scripts/task_tracker.py}"
: "${PANEL:?export PANEL}"
: "${EMBEDDING_ROOT:?export EMBEDDING_ROOT}"

models=(chinese_roberta bge_m3 qwen3_embedding_8b)
matrices=(chinese_roberta/chinese_roberta.npy bge_m3/bge_m3.npy qwen3_embedding_8b/ollama.npy)
metadata=(chinese_roberta/metadata.parquet bge_m3/metadata.parquet qwen3_embedding_8b/metadata.parquet)
labels=(chinese_roberta bge_m3 qwen3_embedding_8b)
model=${models[$SLURM_ARRAY_TASK_ID]}
matrix=${EMBEDDING_ROOT}/${matrices[$SLURM_ARRAY_TASK_ID]}
meta=${EMBEDDING_ROOT}/${metadata[$SLURM_ARRAY_TASK_ID]}
output=${OUTPUT_ROOT:-${EMBEDDING_ROOT}/classification}/${labels[$SLURM_ARRAY_TASK_ID]}_logistic.json

mkdir -p "$(dirname "$output")" logs/slurm_sina_full_classification
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

.venv/bin/python scripts/run_precomputed_embedding_classification.py \
  "${PANEL}" --matrix "${matrix}" --metadata "${meta}" \
  --embedding-model "${model}" --input-name sina_full_cleaned \
  --classifier logistic --reducer pca --reducer-components 128 --scaler auto \
  --train-target-column event_return_3d --evaluation-target-column next_day_return \
  --date-column entry_date --panel-row-index-column row_index \
  --metadata-row-index-column row_index --expected-rows "${EXPECTED_ROWS}" \
  --search-stage coarse --seed 42 --output "${output}"
