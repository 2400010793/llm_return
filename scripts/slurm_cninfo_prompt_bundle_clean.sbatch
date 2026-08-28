#!/bin/bash
# Build one fixed-point-cleaned CNINFO Prompt bundle without the serial audit.
# The parallel audit is submitted as a separate dependency chain.
#SBATCH --job-name=cninfo-prompt-clean
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/home/team/llm_return/logs/slurm_cninfo_prompt_clean/%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_cninfo_prompt_clean/%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/team/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?clean prompt construction must use scripts/task_tracker.py}"
: "${INPUT:?INPUT is required}"
: "${BUNDLE_DIR:?BUNDLE_DIR is required}"

PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
SHORT_DIR=${SHORT_DIR:-${BUNDLE_DIR}/prompt_v3_fixed_parts}
LONG_DIR=${LONG_DIR:-${BUNDLE_DIR}/prompt_v4_fixed_long_parts}
MANIFEST=${MANIFEST:-${BUNDLE_DIR}/manifest.json}
ROW_OFFSET=${ROW_OFFSET:-0}
PART_SIZE=${PART_SIZE:-5000}
NUM_SHARDS=${NUM_SHARDS:-32}

mkdir -p logs/slurm_cninfo_prompt_clean "${BUNDLE_DIR}"
"${PYTHON}" -m py_compile \
  scripts/build_cninfo_prompt_inputs.py scripts/build_cninfo_prompt_bundle.py
"${PYTHON}" scripts/build_cninfo_prompt_bundle.py \
  --input "${INPUT}" \
  --short-output-dir "${SHORT_DIR}" \
  --long-output-dir "${LONG_DIR}" \
  --manifest "${MANIFEST}" \
  --row-offset "${ROW_OFFSET}" \
  --part-size "${PART_SIZE}" \
  --num-shards "${NUM_SHARDS}"
[[ -s "${MANIFEST}" ]]
