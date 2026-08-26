#!/bin/bash
# Assemble one prompt representation after its 64 embedding shards complete.
#SBATCH --job-name=aligned-ms-matrix
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/aligned_masked_short_matrix/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/aligned_masked_short_matrix/%A_%a.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?array task required}"
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
MANIFEST=${MANIFEST:?MANIFEST is required}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id kind axis level prompt_id model variant representation shards embeddings_root output_root <<< "${line%$'\r'}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || { echo "manifest task id mismatch" >&2; exit 2; }
cd "${REPO_ROOT}"
mkdir -p /mnt/lustre3/home/gaozh/llm_return/logs/aligned_masked_short_matrix
.venv/bin/python scripts/build_aligned_prompt_matrix.py \
  --embeddings-root "${embeddings_root}" --model "${model}" \
  --prompt-id "${prompt_id}" --representation "${representation}" \
  --shards "${shards}" --output-root "${output_root}"
