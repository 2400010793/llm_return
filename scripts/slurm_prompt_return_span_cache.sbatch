#!/bin/bash
#SBATCH --job-name=return-span-cache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/return_span_cluster/%A_%a_cache.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/return_span_cluster/%A_%a_cache.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?run as array}"
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/return_span_cluster_v1}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-${SINA_ROOT}/prompt_minimal_v2/embeddings}
EXPECTED_ROWS=${EXPECTED_ROWS:-75894}
SEMANTIC_NAME=${SEMANTIC_NAME:-return_span}
PHRASE=${PHRASE:-收益}
MANIFEST=${CACHE_MANIFEST:-${REPO_ROOT}/configs/generated/minimal_prompt_cache_v2.tsv}
cd "${REPO_ROOT}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id model prompt_length <<< "${line%$'\r'}"
destination=${STUDY_ROOT}/cache/${model}/${prompt_length}
if [[ -s "${destination}/COMPLETED" ]]; then exit 0; fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/build_prompt_return_span_cache.py \
  --embedding-root "${EMBEDDING_ROOT}" \
  --output-root "${STUDY_ROOT}/cache" --model "${model}" \
  --prompt-length "${prompt_length}" --expected-shards 64 --expected-rows "${EXPECTED_ROWS}" \
  --semantic-name "${SEMANTIC_NAME}" --phrase "${PHRASE}"
