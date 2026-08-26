#!/bin/bash
# Build and audit prompts only for 2010--2017 appended rows.
#SBATCH --job-name=cninfo-prompt-increment
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_cninfo_prompt_increment/%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_cninfo_prompt_increment/%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/gaozh/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?incremental prompt construction must use scripts/task_tracker.py}"

PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
INPUT=${INPUT:-data/processed/cleaned/cninfo_announcements_prompt_inputs_2010_2017_increment.jsonl}
BUNDLE_DIR=${BUNDLE_DIR:-data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017}
SHORT_DIR=${SHORT_DIR:-${BUNDLE_DIR}/prompt_v3_fixed_parts}
LONG_DIR=${LONG_DIR:-${BUNDLE_DIR}/prompt_v4_fixed_long_parts}
MANIFEST=${MANIFEST:-${BUNDLE_DIR}/manifest.json}
SUMMARY=${SUMMARY:-reports/cninfo_prompt_bundle_increment_2010_2017_audit.json}
ROW_OFFSET=${ROW_OFFSET:-350577}
PART_SIZE=${PART_SIZE:-5000}
NUM_SHARDS=${NUM_SHARDS:-32}

mkdir -p logs/slurm_cninfo_prompt_increment "${BUNDLE_DIR}" "$(dirname "${SUMMARY}")"
"${PYTHON}" -m py_compile scripts/build_cninfo_prompt_bundle.py scripts/audit_cninfo_prompt_bundle.py
"${PYTHON}" scripts/build_cninfo_prompt_bundle.py \
  --input "${INPUT}" \
  --short-output-dir "${SHORT_DIR}" \
  --long-output-dir "${LONG_DIR}" \
  --manifest "${MANIFEST}" \
  --row-offset "${ROW_OFFSET}" \
  --part-size "${PART_SIZE}" \
  --num-shards "${NUM_SHARDS}"
"${PYTHON}" scripts/audit_cninfo_prompt_bundle.py \
  --input "${INPUT}" \
  --short-dir "${SHORT_DIR}" \
  --long-dir "${LONG_DIR}" \
  --manifest "${MANIFEST}" \
  --summary "${SUMMARY}" \
  --row-offset "${ROW_OFFSET}" \
  --num-shards "${NUM_SHARDS}"
