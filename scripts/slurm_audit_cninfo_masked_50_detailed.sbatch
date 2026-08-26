#!/bin/bash
#SBATCH --job-name=cninfo-audit50
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_audit50/%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_audit50/%j.err
set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_audit50 reports/data_quality
/usr/bin/python3 scripts/audit_cninfo_masked_50_detailed.py \
  --input data/processed/cleaned/cninfo_announcements_final.jsonl \
  --json-output reports/data_quality/cninfo_masked_random50_detailed.json \
  --txt-output reports/data_quality/cninfo_masked_random50_detailed.txt \
  --size 50 --seed 20260807
