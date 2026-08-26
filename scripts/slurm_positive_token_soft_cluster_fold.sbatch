#!/bin/bash
#SBATCH --job-name=positive-token-soft
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/positive_token_soft_cluster/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/positive_token_soft_cluster/%A_%a.err

set -euo pipefail
: "${MANIFEST:?MANIFEST required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT required}"
: "${SLURM_ARRAY_TASK_ID:?array task required}"
REPO=/mnt/lustre3/home/gaozh/llm_return
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id prompt span model variant test_year panel matrix metadata representation projection_mode <<< "${line%$'\r'}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]]
destination=${OUTPUT_ROOT}/folds/${prompt}/${model}/${variant}
representation_args=()
if [[ -n "${representation:-}" ]]; then
  destination=${destination}/${representation}
  representation_args+=(--representation "${representation}")
fi
projection_mode=${projection_mode:-pca}
if [[ "${projection_mode}" == raw ]]; then
  destination=${destination}/raw
fi
destination=${destination}/${test_year}
[[ -s "${destination}/COMPLETED" ]] && exit 0
cd "${REPO}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
.venv/bin/python scripts/run_positive_token_soft_cluster_fold.py \
  --panel "${panel}" --matrix "${matrix}" --metadata "${metadata}" \
  --output-root "${OUTPUT_ROOT}/folds" --prompt "${prompt}" --model "${model}" \
  --variant "${variant}" --test-year "${test_year}" \
  --projection-mode "${projection_mode}" "${representation_args[@]}"
