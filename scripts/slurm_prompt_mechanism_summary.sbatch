#!/bin/bash
#SBATCH --job-name=prompt-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/prompt_mechanism/%j_summary.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/prompt_mechanism/%j_summary.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_mechanism_v1}
cd "${REPO_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
.venv/bin/python scripts/summarize_prompt_mechanism_study.py \
  --input-root "${STUDY_ROOT}/results" --output-root "${STUDY_ROOT}/summary" \
  --bootstrap 2000
