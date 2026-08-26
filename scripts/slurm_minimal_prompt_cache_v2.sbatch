#!/bin/bash
#SBATCH --job-name=min-prompt-cache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/minimal_prompt_knn/%A_%a_cache.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/minimal_prompt_knn/%A_%a_cache.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?run as an array}"
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/analysis_v1}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-${SINA_ROOT}/prompt_minimal_v2/embeddings}
PANEL=${PANEL:-${SINA_ROOT}/classification/sina_single_stock_classification_panel.parquet}
EXPECTED_ROWS=${EXPECTED_ROWS:-75894}
MANIFEST=${CACHE_MANIFEST:-${REPO_ROOT}/configs/generated/minimal_prompt_cache_v2.tsv}
cd "${REPO_ROOT}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id model prompt_length <<< "${line%$'\r'}"
: "${model:?missing model}" "${prompt_length:?missing prompt length}"
destination=${STUDY_ROOT}/cache/${model}/${prompt_length}
if [[ -s "${destination}/COMPLETED" ]]; then
  echo "reusing ${destination}"
  exit 0
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/build_prompt_mechanism_cache.py \
  --embedding-root "${EMBEDDING_ROOT}" \
  --panel "${PANEL}" \
  --output-root "${STUDY_ROOT}/cache" --model "${model}" \
  --prompt-length "${prompt_length}" --prompt-spec configs/prompts/minimal_v2.json \
  --expected-shards 64 --expected-rows "${EXPECTED_ROWS}" --batch-size 64 \
  --without-body --without-topk
