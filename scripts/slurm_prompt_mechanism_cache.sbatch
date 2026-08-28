#!/bin/bash
#SBATCH --job-name=prompt-cache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/prompt_mechanism/%A_%a_cache.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/prompt_mechanism/%A_%a_cache.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?run as an array}"
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_mechanism_v1}
MANIFEST=${CACHE_MANIFEST:-${REPO_ROOT}/configs/generated/prompt_mechanism_cache_v1.tsv}
cd "${REPO_ROOT}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id model prompt_length <<< "${line%$'\r'}"
: "${model:?missing model}" "${prompt_length:?missing prompt length}"
destination=${STUDY_ROOT}/cache/${model}/${prompt_length}
if [[ -s "${destination}/COMPLETED" ]]; then
  echo "reusing completed cache ${destination}"
  exit 0
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/build_prompt_mechanism_cache.py \
  --embedding-root "${SINA_ROOT}/embeddings" \
  --panel "${SINA_ROOT}/classification/sina_single_stock_classification_panel.parquet" \
  --output-root "${STUDY_ROOT}/cache" --model "${model}" \
  --prompt-length "${prompt_length}" --expected-shards 64 --expected-rows 75894 \
  --batch-size 32 --topk 8
