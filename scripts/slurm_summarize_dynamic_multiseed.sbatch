#!/bin/bash
#SBATCH --job-name=summarize-dynamic-ms
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/team/llm_return/logs/slurm_dynamic_prompt/%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_dynamic_prompt/%j.err

set -euo pipefail
REPO_ROOT=/home/team/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?summary must be submitted through scripts/task_tracker.py}"

shopt -s nullglob
reports=(
  reports/dynamic_prompt/seed42/roberta_masked_short_{uniform,static,dynamic}_regression_seed42.json
  reports/dynamic_prompt/multiseed/roberta_masked_short_*_regression_seed*.json
)
[[ ${#reports[@]} -eq 15 ]] || {
  echo "expected 15 completed reports; found ${#reports[@]}" >&2
  exit 2
}

export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}
/usr/bin/python3 scripts/summarize_dynamic_prompt_matrix.py \
  "${reports[@]}" \
  --output reports/dynamic_prompt/multiseed/validation_comparison.json
