#!/bin/bash
#SBATCH --job-name=qwen-pos-reg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/qwen_position/%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/qwen_position/%j.err
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return
: "${MATRIX:?}"; : "${META:?}"; : "${PANEL:?}"; : "${OUTPUT:?}"; : "${FACTOR_ID:?}"
TOP_N=${TOP_N:-100}
mkdir -p "$(dirname "${OUTPUT}")" "${OUTPUT}.simple_states" logs/qwen_position
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
.venv/bin/python scripts/run_positive_prompt_cluster_regression.py "${PANEL}" --matrix "${MATRIX}" --metadata "${META}" --components 128 --output "${OUTPUT}"
.venv/bin/python scripts/evaluate_predictions_simple_states.py "${OUTPUT}" --output-dir "${OUTPUT}.simple_states_top${TOP_N}" --factor-id "${FACTOR_ID}_top${TOP_N}" --label-lag 1 --direction 1 --top-n "${TOP_N}" --min-amount 20000000 --max-missing-weight 0.05 || echo "simple_states returned nonzero; prediction output retained" >&2
