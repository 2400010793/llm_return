#!/bin/bash
#SBATCH --job-name=cninfo-tree-stack
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/cninfo_tree_stacking/%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/cninfo_tree_stacking/%j.err

set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
: "${INPUT_ROOT:?export INPUT_ROOT}"
: "${OUTPUT_ROOT:?export OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" logs/cninfo_tree_stacking
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

.venv/bin/python scripts/run_cninfo_oos_tree_stacking.py \
  --input-root "${INPUT_ROOT}" --output-dir "${OUTPUT_ROOT}" \
  --threads "${SLURM_CPUS_PER_TASK}" --feature-counts 4,8,16,32,72
