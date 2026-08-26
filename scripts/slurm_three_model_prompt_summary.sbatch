#!/bin/bash
#SBATCH --job-name=three-model-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/three_model_prompt_fair/%j_summary.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/three_model_prompt_fair/%j_summary.err
set -euo pipefail
: "${OUTPUT_ROOT:?}" "${REPORT:?}"
cd /mnt/lustre3/home/gaozh/llm_return
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
.venv/bin/python scripts/summarize_three_model_prompt_fair.py \
  --root "${OUTPUT_ROOT}" --report "${REPORT}"
