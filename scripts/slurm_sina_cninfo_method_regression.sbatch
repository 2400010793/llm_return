#!/bin/bash
#SBATCH --job-name=sina-cninfo-reg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_sina_cninfo_method/%A_%a_reg.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_sina_cninfo_method/%A_%a_reg.err

set -euo pipefail
cd /home/gaozh/llm_return
: "${MANIFEST:?export MANIFEST}"
: "${SLURM_ARRAY_TASK_ID:?run as array}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id embedding_root model variant feature target regressor reducer components output <<< "${line%$'\r'}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || exit 2
mkdir -p "$(dirname "${output}")" logs/slurm_sina_cninfo_method
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

.venv/bin/python scripts/run_pooled_embedding_regression.py \
  /home/gaozh/news_content_quality_20260812/classification/sina_single_stock_classification_panel.parquet \
  --embedding-root "${embedding_root}" --model "${model}" --variant "${variant}" \
  --feature "${feature}" --target-column "${target}" --regressor "${regressor}" \
  --reducer "${reducer}" --reducer-components "${components:-128}" \
  --date-column entry_date --stock-column stock_id --panel-row-index-column row_index \
  --expected-embedding-rows 4928 --max-embedding-matrix-gib 12 \
  --search-stage coarse --run-mode final-test --seed 42 --output "${output}"
