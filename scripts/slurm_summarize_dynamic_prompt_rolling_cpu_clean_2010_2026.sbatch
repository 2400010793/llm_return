#!/bin/bash
#SBATCH --job-name=prompt-dynamic-cpu-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_dynamic_prompt_cpu_clean/summary_%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_dynamic_prompt_cpu_clean/summary_%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${MODEL:?export MODEL=roberta or bge_m3}"
ROOT=${ROOT:-reports/dynamic_prompt/rolling_cpu_clean_2010_2026}
OUTPUT=${OUTPUT:-${ROOT}/${MODEL}_masked_short_summary.json}
mkdir -p logs/slurm_dynamic_prompt_cpu_clean
.venv/bin/python scripts/summarize_dynamic_prompt_rolling_cpu_clean_2010_2026.py \
  --root "${ROOT}" --model "${MODEL}" --output "${OUTPUT}"
