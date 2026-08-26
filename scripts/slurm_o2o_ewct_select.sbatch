#!/bin/bash
#SBATCH --job-name=o2o-ewct-select
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_o2o_ewct/%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_o2o_ewct/%j.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?selection job must be tracked}"
: "${MANIFEST:?export MANIFEST}"
: "${SELECTION_OUTPUT:?export SELECTION_OUTPUT}"
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

/usr/bin/python3 scripts/select_o2o_ewct.py \
  --manifest "${MANIFEST}" --output "${SELECTION_OUTPUT}"
