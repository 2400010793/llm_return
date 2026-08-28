#!/bin/bash
#SBATCH --job-name=three-model-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/three_model_prompt_fair/%j_summary.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/three_model_prompt_fair/%j_summary.err
set -euo pipefail
: "${OUTPUT_ROOT:?}" "${REPORT:?}"
cd /data/alpha_team2/shares/llm_return
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
.venv/bin/python scripts/summarize_three_model_prompt_fair.py \
  --root "${OUTPUT_ROOT}" --report "${REPORT}"
