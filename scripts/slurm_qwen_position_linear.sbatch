#!/bin/bash
#SBATCH --job-name=qwen-pos-linear
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/qwen_position/%j.linear.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/qwen_position/%j.linear.err
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return
: "${MATRIX:?}"; : "${META:?}"; : "${PANEL:?}"; : "${OUTPUT:?}"; : "${FACTOR_ID:?}"
mkdir -p "$(dirname "${OUTPUT}")" "${OUTPUT}.simple_states" logs/qwen_position
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
args=(--reducer "${REDUCER:-none}")
if [[ "${REDUCER:-none}" == pca ]]; then args+=(--reducer-components "${COMPONENTS}"); fi
.venv/bin/python scripts/run_sina_precomputed_regression.py "${PANEL}" --matrix "${MATRIX}" --metadata "${META}" --embedding-model qwen3_embedding_8b --variant prompt_short --target-column next_day_return --panel-row-index-column row_index --metadata-row-index-column row_index --regressor ridge --run-mode final-test --expected-rows "${EXPECTED_ROWS}" --search-stage fine --output "${OUTPUT}" "${args[@]}"
pred=$(find "${OUTPUT}" -type f -name '*predictions*.parquet' -print -quit 2>/dev/null || true)
[[ -n "${pred}" ]] || pred="${OUTPUT}"
.venv/bin/python scripts/evaluate_predictions_simple_states.py "${pred}" --output-dir "${OUTPUT}.simple_states" --factor-id "${FACTOR_ID}" --label-lag 1 --direction 1 --top-n 100 --min-amount 20000000 --max-missing-weight 0.05 || echo "simple_states returned nonzero; regression retained" >&2
