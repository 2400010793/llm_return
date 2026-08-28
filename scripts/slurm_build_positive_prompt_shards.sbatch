#!/bin/bash
# Streaming repartition only; no model inference is performed here.
#SBATCH --job-name=positive-prompt-shards
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/positive_prompt_embeddings/%j_shards.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/positive_prompt_embeddings/%j_shards.err
set -euo pipefail
: "${TASK_RECORD_ID:?child task must be recorded}"
REPO_ROOT=${REPO_ROOT:-/home/team/llm_return}
SOURCE=${SOURCE:?SOURCE required}
OUTPUT=${OUTPUT:?OUTPUT required}
cd "${REPO_ROOT}"
if [[ -s "${OUTPUT}/manifest.json" ]]; then exit 0; fi
.venv/bin/python scripts/build_prompt_template_shards.py --source "${SOURCE}" --output "${OUTPUT}" --shards 1000