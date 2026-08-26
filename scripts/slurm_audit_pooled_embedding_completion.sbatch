#!/bin/bash
#SBATCH --job-name=audit-full-embed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_embedding_audit/%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_embedding_audit/%j.err

set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_embedding_audit reports
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
/usr/bin/python3 scripts/audit_pooled_embedding_completion.py \
  --expected-rows 350577 \
  --output reports/embedding_completion_audit.json