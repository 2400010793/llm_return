#!/bin/bash
#SBATCH --job-name=clean-paper-news
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_clean_paper_news/%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_clean_paper_news/%j.err
set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_clean_paper_news data/processed/cleaned
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
/usr/bin/python3 scripts/clean_paper_news.py \
  data/interim/cninfo_paper_1000_focus \
  --output data/processed/cleaned/cninfo_paper_news_clean.parquet \
  --min-chars 500 \
  --max-chars 50000
