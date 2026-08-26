#!/bin/bash
#SBATCH --job-name=prompt-token-clean-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_prompt_token_cls_clean/summary_%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_prompt_token_cls_clean/summary_%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${MODEL:?export MODEL=roberta or bge_m3}"
ROOT=${ROOT:-reports/classification/prompt_tokens_2010_2026_clean_v1}
OUTPUT=${OUTPUT:-reports/classification/prompt_tokens_2010_2026_clean_v1/${MODEL}_summary.json}
mkdir -p logs/slurm_prompt_token_cls_clean
.venv/bin/python scripts/summarize_prompt_token_classification_clean_2010_2026.py \
  --root "${ROOT}" --model "${MODEL}" --output "${OUTPUT}"
