#!/bin/bash
# Audit one historical prompt shard per array task.
#SBATCH --job-name=cninfo-prompt-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-31%32
#SBATCH --output=/home/team/llm_return/logs/slurm_cninfo_prompt_audit/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_cninfo_prompt_audit/%A_%a.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/team/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?parallel prompt audit must use scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?parallel prompt audit requires an array task}"

PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
SOURCE_DIR=${SOURCE_DIR:-data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017/audit_source_shards}
SHORT_DIR=${SHORT_DIR:-data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017/prompt_v3_fixed_parts}
LONG_DIR=${LONG_DIR:-data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017/prompt_v4_fixed_long_parts}
SUMMARY_DIR=${SUMMARY_DIR:-data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017/parallel_audit_shards}
NUM_SHARDS=${NUM_SHARDS:-32}
ROW_OFFSET=${ROW_OFFSET:-350577}

mkdir -p logs/slurm_cninfo_prompt_audit "${SUMMARY_DIR}"
"${PYTHON}" -m py_compile scripts/audit_cninfo_prompt_bundle_parallel.py
"${PYTHON}" scripts/audit_cninfo_prompt_bundle_parallel.py shard \
  --source-dir "${SOURCE_DIR}" \
  --short-dir "${SHORT_DIR}" \
  --long-dir "${LONG_DIR}" \
  --output "${SUMMARY_DIR}/shard-${SLURM_ARRAY_TASK_ID}.json" \
  --shard "${SLURM_ARRAY_TASK_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --row-offset "${ROW_OFFSET}"