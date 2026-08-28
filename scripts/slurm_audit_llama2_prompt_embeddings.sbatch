#!/bin/bash
#SBATCH --job-name=audit-llama2-readout
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/llama2_minimal/%j_audit.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/llama2_minimal/%j_audit.err

set -euo pipefail
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${AUDIT_OUTPUT:?AUDIT_OUTPUT is required}"
: "${EXPECTED_SHARDS:?EXPECTED_SHARDS is required}"
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
PROMPT_SPEC=${PROMPT_SPEC:-configs/prompts/minimal_v2.json}
MODEL_DIRECTORY=${MODEL_DIRECTORY:-llama2_13b}
EXPECTED_FORMAT=${EXPECTED_FORMAT:-llama2_causal_readout_v2}
EXPECTED_ROW_ARG=()
[[ -n "${EXPECTED_ROWS:-}" ]] && EXPECTED_ROW_ARG=(--expected-rows "${EXPECTED_ROWS}")
cd "${REPO_ROOT}"
.venv/bin/python scripts/audit_llama2_prompt_embeddings.py \
  --root "${OUTPUT_ROOT}" \
  --prompt-spec "${PROMPT_SPEC}" \
  --model-directory "${MODEL_DIRECTORY}" \
  --expected-format "${EXPECTED_FORMAT}" \
  --output "${AUDIT_OUTPUT}" \
  --expected-shards "${EXPECTED_SHARDS}" \
  "${EXPECTED_ROW_ARG[@]}" ${ALLOW_SUBSET_ARG:-} --strict
