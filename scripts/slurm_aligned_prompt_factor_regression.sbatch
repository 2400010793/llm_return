#!/bin/bash
# Run one leakage-safe aligned factor regression task.
#SBATCH --job-name=aligned-ms-reg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/aligned_masked_short_regression/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/aligned_masked_short_regression/%A_%a.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?array task required}"
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
MANIFEST=${MANIFEST:?MANIFEST is required}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r kind axis factor model variant representation target regressor panel matrix metadata embed_model panel_key metadata_key expected_rows output <<< "${line%$'\r'}"
cd "${REPO_ROOT}"
mkdir -p /mnt/lustre3/home/gaozh/llm_return/logs/aligned_masked_short_regression
.venv/bin/python scripts/run_sina_precomputed_regression.py "${panel}" \
  --matrix "${matrix}" --metadata "${metadata}" \
  --embedding-model "${embed_model}" --variant masked_short \
  --target-column "${target}" --panel-row-index-column "${panel_key}" \
  --metadata-row-index-column "${metadata_key}" --regressor "${regressor}" \
  --run-mode final-test --search-stage coarse --expected-rows "${expected_rows}" \
  --output "${output}"
