#!/bin/bash
#SBATCH --job-name=min-prompt-base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/minimal_prompt_knn/%j_baseline.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/minimal_prompt_knn/%j_baseline.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/analysis_v1}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-${SINA_ROOT}/prompt_minimal_v2/embeddings}
PROMPT_LENGTHS=${PROMPT_LENGTHS:-short long}
MODELS=${MODELS:-roberta bge_m3}
cd "${REPO_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
length_args=()
for length in ${PROMPT_LENGTHS}; do length_args+=(--prompt-length "${length}"); done
model_args=()
for model in ${MODELS}; do model_args+=(--model "${model}"); done
.venv/bin/python scripts/build_prompt_only_baselines.py \
  --embedding-root "${EMBEDDING_ROOT}" \
  --output-root "${STUDY_ROOT}/prompt_only" \
  --prompt-spec configs/prompts/minimal_v2.json \
  "${model_args[@]}" "${length_args[@]}" --device cpu
