#!/bin/bash
# Paper-style rolling Ridge return prediction on the Sina pooling ablation.
#SBATCH --job-name=sina-pooling-reg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-11%4
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_sina_pooling/%A_%a_reg.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_sina_pooling/%A_%a_reg.err

set -euo pipefail
cd /home/gaozh/llm_return
: "${SLURM_ARRAY_TASK_ID:?run as an array over 0..11}"

models=(chinese_bert chinese_roberta bge_m3 xlm_roberta_large)
poolings=(mean cls max)
model=${models[$((SLURM_ARRAY_TASK_ID / 3))]}
pooling=${poolings[$((SLURM_ARRAY_TASK_ID % 3))]}
data_root=/home/gaozh/news_content_quality_20260812
base=${data_root}/embeddings/single_stock/pooling_ablation_v1/${model}
output=${data_root}/regression/results_pooling_ablation_v1/${model}_${pooling}_ridge_none.json

mkdir -p logs/slurm_sina_pooling "$(dirname "${output}")"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

.venv/bin/python scripts/run_sina_precomputed_regression.py \
  "${data_root}/classification/sina_single_stock_classification_panel.parquet" \
  --matrix "${base}/${pooling}.npy" --metadata "${base}/metadata.parquet" \
  --embedding-model "${model}_${pooling}" --variant plain \
  --regressor ridge --reducer none --target-column next_day_return \
  --date-column entry_date --stock-column stock_id \
  --panel-row-index-column article_id --metadata-row-index-column article_id \
  --expected-rows 4928 --search-stage coarse --seed 42 --output "${output}"
