#!/bin/bash
#SBATCH --job-name=dedup-cninfo
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/team/llm_return/logs/slurm_dedup_cninfo/%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_dedup_cninfo/%j.err
set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_dedup_cninfo data/processed/cleaned
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
/usr/bin/python3 scripts/deduplicate_cninfo_cleaned.py \
  --input-dir data/processed/cleaned/cninfo_by_file \
  --output data/processed/cleaned/cninfo_announcements_deduplicated.jsonl \
  --summary data/processed/cleaned/cninfo_announcements_deduplicated.summary.json
