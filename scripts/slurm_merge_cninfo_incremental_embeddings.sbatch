#!/bin/bash
# Build a zero-copy combined embedding root after historical-only encoding.
#SBATCH --job-name=merge-cninfo-emb
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_merge_cninfo_emb/%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_merge_cninfo_emb/%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/gaozh/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?embedding merge must use scripts/task_tracker.py}"

PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
EXISTING_ROOT=${EXISTING_ROOT:-data/processed/pooled_embeddings_v3}
INCREMENTAL_ROOT=${INCREMENTAL_ROOT:-data/processed/pooled_embeddings_increment_2010_2017_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/processed/pooled_embeddings_2010_2026_v1}
EXISTING_ROWS=${EXISTING_ROWS:-350577}
INCREMENTAL_INPUT=${INCREMENTAL_INPUT:-data/processed/cleaned/cninfo_announcements_prompt_inputs_2010_2017_increment.jsonl}
INCREMENTAL_ROWS=${INCREMENTAL_ROWS:-$(wc -l < "${INCREMENTAL_INPUT}")}
MODELS=${MODELS:-roberta,bge_m3}
VARIANTS=${VARIANTS:-short,masked_short}

mkdir -p logs/slurm_merge_cninfo_emb
"${PYTHON}" -m py_compile scripts/merge_incremental_pooled_embeddings.py
"${PYTHON}" scripts/merge_incremental_pooled_embeddings.py \
  --existing-root "${EXISTING_ROOT}" \
  --incremental-root "${INCREMENTAL_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --existing-rows "${EXISTING_ROWS}" \
  --incremental-rows "${INCREMENTAL_ROWS}" \
  --models "${MODELS}" \
  --variants "${VARIANTS}"
