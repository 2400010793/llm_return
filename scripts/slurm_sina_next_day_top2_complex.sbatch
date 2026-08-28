#!/bin/bash
#SBATCH --job-name=sina-top2-complex
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-9%5
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_sina_top2_complex/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_sina_top2_complex/%A_%a.err

set -euo pipefail

REPO_ROOT=/data/alpha_team2/shares/llm_return
DATA_ROOT=/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812
PANEL=${DATA_ROOT}/classification/sina_single_stock_classification_panel.parquet
EMBEDDING_ROOT=${DATA_ROOT}/embeddings/single_stock
OUTPUT_ROOT=${DATA_ROOT}/classification/results_next_day_top2_complex

cd "${REPO_ROOT}"
mkdir -p logs/slurm_sina_top2_complex "${OUTPUT_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export LOKY_MAX_CPU_COUNT=${SLURM_CPUS_PER_TASK}
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

models=(bge_m3 word2vec)
matrices=(bge_m3.npy word2vec.npy)
components=(128 64)
classifiers=(mlp random_forest extra_trees hist_gradient_boosting knn)

model_index=$((SLURM_ARRAY_TASK_ID / 5))
classifier_index=$((SLURM_ARRAY_TASK_ID % 5))
model=${models[$model_index]}
matrix=${matrices[$model_index]}
component=${components[$model_index]}
classifier=${classifiers[$classifier_index]}
output=${OUTPUT_ROOT}/${model}_${classifier}_pca${component}.json

echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
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
