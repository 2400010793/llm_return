#!/bin/bash
#SBATCH --job-name=audit-cninfo-mask
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_audit_cninfo_mask/%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_audit_cninfo_mask/%j.err
set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_audit_cninfo_mask reports/data_quality
/usr/bin/python3 scripts/audit_cninfo_masked_sample.py \
  --input data/processed/cleaned/cninfo_announcements_final.jsonl \
  --output reports/data_quality/cninfo_masked_random50.json \
  --size 50 --seed 20260806
