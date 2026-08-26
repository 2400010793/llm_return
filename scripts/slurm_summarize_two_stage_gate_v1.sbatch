#!/bin/bash
#SBATCH --job-name=summary-gate-v1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_two_stage_v1/summary_%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_two_stage_v1/summary_%j.err

set -euo pipefail
cd /home/gaozh/llm_return
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
export PYTHONPATH="/home/gaozh/llm_return/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}
/usr/bin/python3 scripts/summarize_two_stage_gate_v1.py
