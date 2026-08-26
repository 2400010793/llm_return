#!/bin/bash
#SBATCH --job-name=stock-graph-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_stock_graph/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_stock_graph/%A_%a.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${MANIFEST:?export the graph manifest}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"

DATASET_ROOT=${DATASET_ROOT:-data/processed/stock_graph_features_roberta_masked_v1}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id mode test_year output <<< "${line}"
if [[ -z "${line}" || "${task_id}" != "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "Manifest task mismatch for ${SLURM_ARRAY_TASK_ID}: ${line}" >&2
  exit 2
fi

mkdir -p logs/slurm_stock_graph "$(dirname "${output}")"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

/usr/bin/python3 scripts/run_stock_graph_rolling_classification.py \
  --dataset-root "${DATASET_ROOT}" \
  --mode "${mode}" \
  --test-year "${test_year}" \
  --hidden-size 64 \
  --dropout 0.1 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --max-epochs 30 \
  --patience 4 \
  --batch-size 2048 \
  --gradient-clip-norm 1.0 \
  --num-threads "${SLURM_CPUS_PER_TASK}" \
  --seed 42 \
  --output "${output}"
