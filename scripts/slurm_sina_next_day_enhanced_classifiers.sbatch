#!/bin/bash
#SBATCH --job-name=sina-next1-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-4%4
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_sina_next_day_classification/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_sina_next_day_classification/%A_%a.err

set -euo pipefail

REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
DATA_ROOT=/home/gaozh/news_content_quality_20260812
PANEL=${DATA_ROOT}/classification/sina_single_stock_classification_panel.parquet
EMBEDDING_ROOT=${DATA_ROOT}/embeddings/single_stock
OUTPUT_ROOT=${DATA_ROOT}/classification/results_next_day_all

cd "${REPO_ROOT}"
mkdir -p logs/slurm_sina_next_day_classification "${OUTPUT_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export LOKY_MAX_CPU_COUNT=${SLURM_CPUS_PER_TASK}
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

models=(
  word2vec
  chinese_roberta
  bge_m3
  qwen3_embedding_8b
  qwen3_embedding_8b
)
matrices=(
  word2vec.npy
  chinese_roberta.npy
  bge_m3.npy
  ollama.npy
  ollama.npy
)
classifiers=(
  hist_gradient_boosting
  hist_gradient_boosting
  hist_gradient_boosting
  hist_gradient_boosting
  extra_trees
)
components=(64 128 128 128 128)

task=${SLURM_ARRAY_TASK_ID}
model=${models[$task]}
matrix=${matrices[$task]}
classifier=${classifiers[$task]}
component=${components[$task]}
output=${OUTPUT_ROOT}/${model}_${classifier}_pca${component}.json

echo "job=${SLURM_JOB_ID} task=${task} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
echo "model=${model} classifier=${classifier} pca=${component} output=${output}"

exec .venv/bin/python scripts/run_precomputed_embedding_classification.py \
  "${PANEL}" \
  --matrix "${EMBEDDING_ROOT}/${model}/${matrix}" \
  --metadata "${EMBEDDING_ROOT}/${model}/metadata.parquet" \
  --embedding-model "${model}" \
  --input-name sina_single_stock_text \
  --classifier "${classifier}" \
  --reducer pca \
  --reducer-components "${component}" \
  --scaler auto \
  --train-target-column next_day_return \
  --evaluation-target-column next_day_return \
  --date-column entry_date \
  --panel-row-index-column article_id \
  --metadata-row-index-column article_id \
  --expected-rows 4928 \
  --search-stage coarse \
  --seed 42 \
  --output "${output}"
