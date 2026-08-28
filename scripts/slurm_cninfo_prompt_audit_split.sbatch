#!/bin/bash
# Partition the historical prompt input once for the 32-way audit.
#SBATCH --job-name=cninfo-prompt-audit-split
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=/home/team/llm_return/logs/slurm_cninfo_prompt_audit_split/%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_cninfo_prompt_audit_split/%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/team/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?parallel prompt audit must use scripts/task_tracker.py}"

PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
INPUT=${INPUT:-data/processed/cleaned/cninfo_announcements_prompt_inputs_2010_2017_increment.jsonl}
SOURCE_DIR=${SOURCE_DIR:-data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017/audit_source_shards}
SOURCE_MANIFEST=${SOURCE_MANIFEST:-${SOURCE_DIR}/manifest.json}
NUM_SHARDS=${NUM_SHARDS:-32}
ROW_OFFSET=${ROW_OFFSET:-350577}

mkdir -p logs/slurm_cninfo_prompt_audit_split "${SOURCE_DIR}"
"${PYTHON}" -m py_compile scripts/audit_cninfo_prompt_bundle_parallel.py
"${PYTHON}" scripts/audit_cninfo_prompt_bundle_parallel.py split \
  --input "${INPUT}" \
  --output-dir "${SOURCE_DIR}" \
  --manifest "${SOURCE_MANIFEST}" \
  --num-shards "${NUM_SHARDS}" \
  --row-offset "${ROW_OFFSET}"
[[ -s "${SOURCE_MANIFEST}" ]]