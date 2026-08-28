#!/bin/bash
#SBATCH --job-name=summarize-overnight-v1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/team/llm_return/logs/slurm_overnight_v1/summary_%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_overnight_v1/summary_%j.err

set -euo pipefail
cd /home/team/llm_return
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
/usr/bin/python3 scripts/summarize_overnight_v1.py
