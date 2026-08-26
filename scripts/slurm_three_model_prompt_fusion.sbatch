#!/bin/bash
#SBATCH --job-name=three-model-fusion
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/three_model_prompt_fair/%A_%a_fusion.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/three_model_prompt_fair/%A_%a_fusion.err
set -euo pipefail
: "${OUTPUT_ROOT:?}" "${FIRST_TEST_YEAR:?}"
cd /mnt/lustre3/home/gaozh/llm_return
test_year=$((FIRST_TEST_YEAR + SLURM_ARRAY_TASK_ID))
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
.venv/bin/python scripts/run_three_model_prompt_fusion.py \
  --root "${OUTPUT_ROOT}" --test-year "${test_year}" \
  --output "${OUTPUT_ROOT}/fusion/${test_year}"
