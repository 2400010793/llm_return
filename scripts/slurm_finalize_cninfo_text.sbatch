#!/bin/bash
#SBATCH --job-name=final-cninfo-text
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_finalize_cninfo/%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_finalize_cninfo/%j.err
set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_finalize_cninfo data/processed/cleaned
/usr/bin/python3 scripts/finalize_cninfo_text.py \
  --input data/processed/cleaned/cninfo_announcements_deduplicated.jsonl \
  --output data/processed/cleaned/cninfo_announcements_final.jsonl \
  --excluded data/processed/cleaned/cninfo_announcements_excluded_noise.jsonl \
  --summary data/processed/cleaned/cninfo_announcements_final.summary.json
