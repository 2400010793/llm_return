#!/bin/bash
set -euo pipefail
: "${TASK_RECORD_ID:?run through scripts/task_tracker.py}"
REPO_ROOT=${REPO_ROOT:-/home/team/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
CNINFO_SOURCE=${CNINFO_SOURCE:-/home/team/llm_return/data/processed/cleaned/prompt_v2_masked_parts}
CNINFO_SHARDS_ROOT=${CNINFO_SHARDS_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/cninfo_shards_1000}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/embeddings}
PROMPT_CONFIG=${PROMPT_CONFIG:-${REPO_ROOT}/configs/generated/positive_prompt_templates_v1.json}
mkdir -p "${OUTPUT_ROOT}" /data/alpha_team2/shares/llm_return/logs/positive_prompt_embeddings

tracked() { "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/task_tracker.py" child-submit --task-id "${TASK_RECORD_ID}" -- "$@"; }

# Existing Sina input already has stable short/masked components and 64 shards.
for model in roberta bge_m3; do
  tracked sbatch --parsable --array=0-63%16 \
    --export="ALL,REPO_ROOT=${REPO_ROOT},DATASET=sina,MODEL=${model},SHARDS=64,INPUT_ROOT=${SINA_ROOT}/inputs,OUTPUT_ROOT=${OUTPUT_ROOT}/sina,PROMPT_CONFIG=${PROMPT_CONFIG}" \
    "${REPO_ROOT}/scripts/slurm_prompt_template_embeddings.sbatch"
done

# CNINFO is first repartitioned into 1000 deterministic pieces to avoid the
# historical long single-shard timeout.  The embedding arrays depend on it.
shard_job=$(tracked sbatch --parsable \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SOURCE=${CNINFO_SOURCE},OUTPUT=${CNINFO_SHARDS_ROOT}" \
  "${REPO_ROOT}/scripts/slurm_build_positive_prompt_shards.sbatch")
for model in roberta bge_m3; do
  tracked sbatch --parsable --dependency="afterok:${shard_job}" --array=0-999%32 \
    --export="ALL,REPO_ROOT=${REPO_ROOT},DATASET=cninfo,MODEL=${model},SHARDS=1000,INPUT_ROOT=${CNINFO_SHARDS_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}/cninfo,PROMPT_CONFIG=${PROMPT_CONFIG}" \
    "${REPO_ROOT}/scripts/slurm_prompt_template_embeddings.sbatch"
done
echo "cninfo_shard_job=${shard_job} output_root=${OUTPUT_ROOT}"