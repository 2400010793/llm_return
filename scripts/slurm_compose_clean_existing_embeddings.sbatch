#!/bin/bash
# Compose frozen unmasked short embeddings with newly cleaned masked embeddings.
#SBATCH --job-name=compose-clean-emb
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_compose_clean_emb/%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_compose_clean_emb/%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/gaozh/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?embedding composition must use scripts/task_tracker.py}"

PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
SHORT_ROOT=${SHORT_ROOT:-data/processed/pooled_embeddings_v3}
MASKED_ROOT=${MASKED_ROOT:-data/processed/pooled_embeddings_existing_2018_2026_cleanmask_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/processed/pooled_embeddings_existing_2018_2026_clean_v1}
MODELS=${MODELS:-roberta,bge_m3}

mkdir -p logs/slurm_compose_clean_emb
"${PYTHON}" -m py_compile scripts/compose_pooled_embedding_variants.py
"${PYTHON}" scripts/compose_pooled_embedding_variants.py \
  --short-root "${SHORT_ROOT}" \
  --masked-root "${MASKED_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --models "${MODELS}"
