#!/bin/bash
#SBATCH --job-name=qwen-cluster-linear
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_qwen_linear/%j_cluster.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_qwen_linear/%j_cluster.err

set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
FIT_WINDOW_YEARS=${FIT_WINDOW_YEARS:-6}
VALIDATION_WINDOW_YEARS=${VALIDATION_WINDOW_YEARS:-2}
OUTPUT_ROOT=${OUTPUT_ROOT:?export OUTPUT_ROOT}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
.venv/bin/python scripts/run_qwen_cluster_linear.py \
  --panel data/processed/cninfo_full_o2o_residual_panel.parquet \
  --embedding-root data/processed/pooled_embeddings_qwen_candidates_1000 \
  --selected reports/regression/qwen_2018_2026_linear_v1/selected_candidates.json \
  --output-root "${OUTPUT_ROOT}" --fit-window-years "${FIT_WINDOW_YEARS}" \
  --validation-window-years "${VALIDATION_WINDOW_YEARS}"
