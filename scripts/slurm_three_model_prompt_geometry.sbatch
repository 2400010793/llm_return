#!/bin/bash
#SBATCH --job-name=three-model-geometry
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/three_model_prompt_fair/%j_geometry.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/three_model_prompt_fair/%j_geometry.err
set -euo pipefail
: "${OUTPUT_ROOT:?}"
cd /mnt/lustre3/home/gaozh/llm_return
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
.venv/bin/python scripts/analyze_three_model_prompt_geometry.py --root "${OUTPUT_ROOT}"
