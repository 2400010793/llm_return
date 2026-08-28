#!/bin/bash
#SBATCH --job-name=clean-cninfo
#SBATCH --partition=cpu
#SBATCH --array=0-967%16
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home/team/llm_return/logs/slurm_clean_cninfo/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_clean_cninfo/%A_%a.err
set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_clean_cninfo data/processed/cleaned/cninfo_by_file
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
mapfile -t files < <(find data/interim/cninfo_paper_1000_focus -maxdepth 1 -type f -name '*.json' -printf '%p\n' | sort)
input="${files[$SLURM_ARRAY_TASK_ID]}"
if [[ -z "${input:-}" ]]; then
  echo "No input for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi
/usr/bin/python3 scripts/clean_cninfo_one_file.py \
  "$input" \
  --output-dir data/processed/cleaned/cninfo_by_file
