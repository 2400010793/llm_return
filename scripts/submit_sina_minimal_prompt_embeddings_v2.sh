#!/bin/bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
PROMPT_SPEC=${PROMPT_SPEC:-${REPO_ROOT}/configs/prompts/minimal_v2.json}
CONCURRENCY=${CONCURRENCY:-64}
DATASET=${DATASET:-sina}
if [[ "${DATASET}" == "cninfo" ]]; then
  OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/data/processed/cninfo_minimal_prompt_v2/embeddings}
  AUDIT_OUTPUT=${AUDIT_OUTPUT:-${REPO_ROOT}/reports/cninfo_minimal_prompt_v2/embedding_audit.json}
  EXPECTED_ROWS=${EXPECTED_ROWS:-903665}
  ARRAY_END=255
  VARIANTS="short masked_short"
else
  OUTPUT_ROOT=${OUTPUT_ROOT:-${SINA_ROOT}/prompt_minimal_v2/embeddings}
  AUDIT_OUTPUT=${AUDIT_OUTPUT:-${SINA_ROOT}/prompt_minimal_v2/embedding_audit.json}
  EXPECTED_ROWS=${EXPECTED_ROWS:-75894}
  ARRAY_END=511
  VARIANTS="short masked_short long masked_long"
fi
cd "${REPO_ROOT}"
TOKENIZER_AUDIT=${TOKENIZER_AUDIT:-$(dirname "${AUDIT_OUTPUT}")/tokenizer_audit.json}
mkdir -p logs/minimal_prompt_v2 "$(dirname "${AUDIT_OUTPUT}")"
.venv/bin/python scripts/build_minimal_prompt_token_audit.py \
  --spec "${PROMPT_SPEC}" \
  --output "${TOKENIZER_AUDIT}" \
  --models roberta bge_m3 > logs/minimal_prompt_v2/tokenizer_audit.out
common="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},PROMPT_SPEC=${PROMPT_SPEC},DATASET=${DATASET},EXPECTED_ROWS=${EXPECTED_ROWS},AUDIT_OUTPUT=${AUDIT_OUTPUT},VARIANTS=${VARIANTS}"
embedding_job=$(sbatch --parsable --array="0-${ARRAY_END}%${CONCURRENCY}" \
  --export="${common}" scripts/slurm_sina_minimal_prompt_embeddings_v2.sbatch)
audit_job=$(sbatch --parsable --dependency="afterany:${embedding_job}" \
  --export="${common}" scripts/slurm_audit_sina_minimal_prompt_embeddings_v2.sbatch)
echo "minimal_prompt_embeddings=${embedding_job} tasks=$((ARRAY_END + 1)) concurrency=${CONCURRENCY} audit=${audit_job} dataset=${DATASET}"
echo "output_root=${OUTPUT_ROOT}"
