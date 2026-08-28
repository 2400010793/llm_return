#!/bin/bash
#SBATCH --job-name=paper-hk-fetch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/home/team/llm_return/logs/slurm_paper_hk/%j_fetch.out
#SBATCH --error=/home/team/llm_return/logs/slurm_paper_hk/%j_fetch.err

set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_paper_hk models
export HF_HOME=/home/team/.cache/huggingface
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
/usr/bin/python3 scripts/prefetch_paper_hk_models.py \
  --model-root models \
  --models xlm_roberta_large
