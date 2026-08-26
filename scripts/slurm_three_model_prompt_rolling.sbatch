#!/bin/bash
#SBATCH --job-name=three-model-rolling
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/three_model_prompt_fair/%A_%a_rolling.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/three_model_prompt_fair/%A_%a_rolling.err
set -euo pipefail
: "${OUTPUT_ROOT:?}"
cd /mnt/lustre3/home/gaozh/llm_return
manifest=${OUTPUT_ROOT}/intersection/fold_manifest.tsv
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${manifest}")
IFS=$'\t' read -r ignored config_id model prompt phrase representation matrix metadata test_year history_years <<< "${line%$'\r'}"
output=${OUTPUT_ROOT}/rolling/${model}/${prompt}/${representation}/${test_year}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
.venv/bin/python scripts/run_three_model_prompt_fair_fold.py \
  --panel "${OUTPUT_ROOT}/intersection/panel.parquet" --matrix "${matrix}" \
  --metadata "${metadata}" --model "${model}" --prompt "${prompt}" \
  --representation "${representation}" --test-year "${test_year}" \
  --history-years "${history_years}" --validation-years "${VALIDATION_YEARS:-1}" \
  --output "${output}"
