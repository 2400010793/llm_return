#!/bin/bash
#SBATCH --job-name=finbert2-fetch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_finbert2/%j_fetch.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_finbert2/%j_fetch.err

set -euo pipefail
cd /data/alpha_team2/shares/llm_return
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
mkdir -p logs/slurm_finbert2 models
export HF_HOME=/mnt/lustre3/home/team/.cache/huggingface
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/data/alpha_team2/shares/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
/usr/bin/python3 scripts/prefetch_finbert2_models.py \
  --model-root models --models finbert2_base --retries 2 --max-workers 2
