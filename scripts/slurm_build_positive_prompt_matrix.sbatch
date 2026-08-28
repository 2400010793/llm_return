#!/bin/bash
#SBATCH --job-name=positive-matrix
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/positive_prompt_matrix/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/positive_prompt_matrix/%A_%a.err
set -euo pipefail
: "${MANIFEST:?MANIFEST required}"
: "${EMBEDDINGS_ROOT:?EMBEDDINGS_ROOT required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT required}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
IFS=$'\t' read -r dataset prompt model variant representation shards <<< "${line%$'\r'}"
cd /data/alpha_team2/shares/llm_return
.venv/bin/python scripts/build_positive_prompt_matrix.py \
  --embeddings-root "$EMBEDDINGS_ROOT" --dataset "$dataset" --prompt "$prompt" \
  --model "$model" --variant "$variant" --representation "$representation" \
  --output-root "$OUTPUT_ROOT" --shards "$shards"
