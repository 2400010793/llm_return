#!/bin/bash
#SBATCH --job-name=qwen-sina-pca
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/qwen_sina_pca/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/qwen_sina_pca/%A_%a.err

set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
: "${MANIFEST:?export MANIFEST}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id variant representation reducer components panel matrix metadata output <<< "${line%$'\r'}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || exit 2
mkdir -p "$(dirname "${output}")" logs/qwen_sina_pca
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

regression_variant=prompt_short
[[ "${variant}" == masked_short ]] && regression_variant=masked_short
args=(--reducer "${reducer}")
[[ "${reducer}" == pca ]] && args+=(--reducer-components "${components}")
.venv/bin/python scripts/run_sina_precomputed_regression.py "${panel}" \
  --matrix "${matrix}" --metadata "${metadata}" \
  --embedding-model qwen3_embedding_8b --variant "${regression_variant}" \
  --target-column next_day_return --panel-row-index-column row_index \
  --metadata-row-index-column row_index --metadata-row-index-offset 0 \
  --regressor ridge --run-mode final-test --expected-rows 75894 \
  --search-stage fine --output "${output}" "${args[@]}"
