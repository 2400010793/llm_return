#!/bin/bash
set -euo pipefail
: "${TASK_RECORD_ID:?run through scripts/task_tracker.py}"
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
CONFIG=${CONFIG:-${REPO_ROOT}/configs/prompts/aligned_masked_short_v1.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-${SINA_ROOT}/aligned_masked_short_v1}
PREFLIGHT=${PREFLIGHT:-${REPO_ROOT}/reports/aligned_masked_short_roberta_preflight.json}

mkdir -p "$(dirname "${PREFLIGHT}")"
.venv/bin/python "${REPO_ROOT}/scripts/audit_aligned_prompt_tokens.py" \
  --config "${CONFIG}" --model roberta \
  --input "${SINA_ROOT}/inputs/shard-0" --max-length 512 --output "${PREFLIGHT}"

tracked() {
  "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/task_tracker.py" child-submit \
    --task-id "${TASK_RECORD_ID}" -- "$@"
}

tracked sbatch --parsable --array=0-255%256 \
  --export="ALL,REPO_ROOT=${REPO_ROOT},INPUT_ROOT=${SINA_ROOT}/inputs,OUTPUT_ROOT=${OUTPUT_ROOT},CONFIG=${CONFIG}" \
  "${REPO_ROOT}/scripts/slurm_aligned_masked_short_embeddings.sbatch"
