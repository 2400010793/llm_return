#!/bin/bash
#SBATCH --job-name=audit-cninfo
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/team/llm_return/logs/slurm_audit_cninfo/%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_audit_cninfo/%j.err
set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_audit_cninfo reports/data_quality
/usr/bin/python3 scripts/audit_cninfo_random_sample.py \
  --input data/processed/cleaned/cninfo_announcements_deduplicated.jsonl \
  --output reports/data_quality/cninfo_deduplicated_random50.txt \
  --size 50 --seed 20260806
