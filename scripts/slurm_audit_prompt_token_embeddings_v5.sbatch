#!/bin/bash
#SBATCH --job-name=audit-prompt-v5
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_prompt_token_v5/%j.audit.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_prompt_token_v5/%j.audit.err
set -euo pipefail
cd /home/gaozh/llm_return
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
/usr/bin/python3 scripts/audit_prompt_token_embeddings_v5.py