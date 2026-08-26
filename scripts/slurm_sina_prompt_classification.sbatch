#!/bin/bash
# Classify every completed non-Qwen Sina prompt/mask embedding.
# Array mapping: 4 embedding models x 5 text variants.
#SBATCH --job-name=sina-prompt-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-19%16
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_sina_prompt_classification/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_sina_prompt_classification/%A_%a.err

set -euo pipefail

REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
DATA_ROOT=/home/gaozh/news_content_quality_20260812
PANEL=${PANEL:-${DATA_ROOT}/classification/sina_single_stock_classification_panel.parquet}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-${DATA_ROOT}/embeddings/single_stock}
OUTPUT_ROOT=${OUTPUT_ROOT:-${DATA_ROOT}/classification/results_prompt_nonqwen}
CLASSIFIER=${CLASSIFIER:-hist_gradient_boosting}
REDUCER=${REDUCER:-none}
COMPONENTS=${COMPONENTS:-128}

cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?classification array must be submitted through scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?run as an array}"
mkdir -p logs/slurm_sina_prompt_classification "${OUTPUT_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export LOKY_MAX_CPU_COUNT=${SLURM_CPUS_PER_TASK}
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

models=(chinese_bert chinese_roberta bge_m3 word2vec)
model_dirs=(chinese_bert_prompt chinese_roberta_prompt bge_m3_prompt word2vec_prompt)
matrix_names=(chinese_bert.npy chinese_roberta.npy bge_m3.npy word2vec.npy)
variants=(plain prompt_short prompt_long masked_short masked_long)
model_index=$((SLURM_ARRAY_TASK_ID / 5))
variant_index=$((SLURM_ARRAY_TASK_ID % 5))
model=${models[$model_index]}
model_dir=${model_dirs[$model_index]}
matrix_name=${matrix_names[$model_index]}
variant=${variants[$variant_index]}
base="${EMBEDDING_ROOT}/${model_dir}/${variant}"
output="${OUTPUT_ROOT}/${model}_${variant}_${CLASSIFIER}_${REDUCER}.json"

echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} model=${model} variant=${variant} classifier=${CLASSIFIER} reducer=${REDUCER}"

exec "${REPO_ROOT}/.venv/bin/python" scripts/run_precomputed_embedding_classification.py \
  "${PANEL}" \
  --matrix "${base}/${matrix_name}" \
  --metadata "${base}/metadata.parquet" \
  --embedding-model "${model}" \
  --input-name "sina_${variant}" \
  --classifier "${CLASSIFIER}" \
  --reducer "${REDUCER}" \
  --reducer-components "${COMPONENTS}" \
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
