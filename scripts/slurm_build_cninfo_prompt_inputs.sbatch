#!/bin/bash
#SBATCH --job-name=cninfo-prompts
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/team/llm_return/logs/slurm_cninfo_prompts/%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_cninfo_prompts/%j.err
set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_cninfo_prompts data/processed/cleaned/prompt_v2_masked_parts
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
/usr/bin/python3 scripts/build_cninfo_masked_parts.py \
  --input data/processed/cleaned/cninfo_announcements_final.jsonl \
  --output-dir data/processed/cleaned/prompt_v2_masked_parts \
  --manifest data/processed/cleaned/cninfo_announcements_prompt_inputs_v2_masked.manifest.json \
  --part-size 5000
