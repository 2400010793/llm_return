#!/bin/bash
# Reduce the 32 independent prompt-audit summaries.
#SBATCH --job-name=cninfo-prompt-audit-reduce
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_cninfo_prompt_audit_reduce/%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_cninfo_prompt_audit_reduce/%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/gaozh/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?parallel prompt audit must use scripts/task_tracker.py}"

PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
SOURCE_MANIFEST=${SOURCE_MANIFEST:-data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017/audit_source_shards/manifest.json}
BUNDLE_MANIFEST=${BUNDLE_MANIFEST:-data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017/manifest.json}
SHORT_DIR=${SHORT_DIR:-data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017/prompt_v3_fixed_parts}
LONG_DIR=${LONG_DIR:-data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017/prompt_v4_fixed_long_parts}
SUMMARY_DIR=${SUMMARY_DIR:-data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017/parallel_audit_shards}
SUMMARY=${SUMMARY:-reports/cninfo_prompt_bundle_increment_2010_2017_audit_parallel.json}
NUM_SHARDS=${NUM_SHARDS:-32}
ROW_OFFSET=${ROW_OFFSET:-350577}

mkdir -p logs/slurm_cninfo_prompt_audit_reduce "$(dirname "${SUMMARY}")"
"${PYTHON}" -m py_compile scripts/audit_cninfo_prompt_bundle_parallel.py
"${PYTHON}" scripts/audit_cninfo_prompt_bundle_parallel.py reduce \
  --source-manifest "${SOURCE_MANIFEST}" \
  --bundle-manifest "${BUNDLE_MANIFEST}" \
  --short-dir "${SHORT_DIR}" \
  --long-dir "${LONG_DIR}" \
  --shard-summary-dir "${SUMMARY_DIR}" \
  --output "${SUMMARY}" \
  --num-shards "${NUM_SHARDS}" \
  --row-offset "${ROW_OFFSET}"
[[ -s "${SUMMARY}" ]]