#!/bin/bash
#SBATCH --job-name=sina-cninfo-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/home/team/llm_return/logs/slurm_sina_cninfo_method/%A_%a_cls.out
#SBATCH --error=/home/team/llm_return/logs/slurm_sina_cninfo_method/%A_%a_cls.err

set -euo pipefail
cd /home/team/llm_return
: "${MANIFEST:?export MANIFEST}"
: "${SLURM_ARRAY_TASK_ID:?run as array}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id embedding_root model variant feature classifier reducer components output train_target evaluation_target <<< "${line%$'\r'}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || exit 2
mkdir -p "$(dirname "${output}")" logs/slurm_sina_cninfo_method
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

.venv/bin/python scripts/run_pooled_embedding_classification.py \
  /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/classification/sina_single_stock_classification_panel.parquet \
  --embedding-root "${embedding_root}" --model "${model}" --variant "${variant}" \
  --feature "${feature}" --classifier "${classifier}" --reducer "${reducer}" \
  --reducer-components "${components:-128}" --scaler auto \
  --train-target-column "${train_target}" --evaluation-target-column "${evaluation_target}" \
  --date-column entry_date --panel-row-index-column row_index \
  --expected-embedding-rows 4928 --max-embedding-matrix-gib 12 \
  --search-stage coarse --run-mode final-test --seed 42 --output "${output}"
