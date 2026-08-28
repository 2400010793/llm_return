#!/bin/bash
#SBATCH --job-name=audit-minimal-prompt-v2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/minimal_prompt_v2/%j_audit.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/minimal_prompt_v2/%j_audit.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-${SINA_ROOT}/prompt_minimal_v2/embeddings}
PROMPT_SPEC=${PROMPT_SPEC:-${REPO_ROOT}/configs/prompts/minimal_v2.json}
EXPECTED_ROWS=${EXPECTED_ROWS:-75894}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-${SINA_ROOT}/prompt_minimal_v2/embedding_audit.json}
VARIANTS=${VARIANTS:-short masked_short long masked_long}
MODELS=${MODELS:-roberta bge_m3}
cd "${REPO_ROOT}"
.venv/bin/python scripts/audit_minimal_prompt_embeddings_v2.py \
  --root "${OUTPUT_ROOT}" --spec "${PROMPT_SPEC}" \
  --output "${AUDIT_OUTPUT}" \
  --expected-shards 64 --expected-rows "${EXPECTED_ROWS}" \
  --models ${MODELS} --variants ${VARIANTS}
