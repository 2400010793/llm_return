#!/bin/bash
#SBATCH --job-name=sina-span-cache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/sina_semantic_span/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/sina_semantic_span/%A_%a.err
set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
MANIFEST=${MANIFEST:-${REPO_ROOT}/configs/generated/sina_semantic_span_cache.tsv}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/semantic_span_v1}
cd "${REPO_ROOT}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id model prompt_length semantic_name phrase <<< "${line%$'\r'}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || exit 2
destination="${STUDY_ROOT}/cache/${model}/${prompt_length}/${semantic_name}"
if [[ -s "${destination}/COMPLETED" ]]; then exit 0; fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/build_prompt_return_span_cache.py \
  --embedding-root "${SINA_ROOT}/prompt_minimal_v2/embeddings" \
  --output-root "${STUDY_ROOT}/cache" --model "${model}" \
  --prompt-length "${prompt_length}" --expected-shards 64 --expected-rows 75894 \
  --semantic-name "${semantic_name}" --phrase "${phrase}" --batch-size 512
