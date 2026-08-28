#!/bin/bash
#SBATCH --job-name=prompt-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/prompt_mechanism/%j_summary.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/prompt_mechanism/%j_summary.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_mechanism_v1}
cd "${REPO_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
.venv/bin/python scripts/summarize_prompt_mechanism_study.py \
  --input-root "${STUDY_ROOT}/results" --output-root "${STUDY_ROOT}/summary" \
  --bootstrap 2000
