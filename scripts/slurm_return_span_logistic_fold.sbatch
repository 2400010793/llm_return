#!/bin/bash
#SBATCH --job-name=span-logistic
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/return_span_logistic/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/return_span_logistic/%A_%a.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?run as an array}"
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
MECHANISM_ROOT=${MECHANISM_ROOT:-${SINA_ROOT}/prompt_minimal_v2/analysis_v1}
SPAN_ROOT=${SPAN_ROOT:-${SINA_ROOT}/prompt_minimal_v2/return_span_cluster_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-${SINA_ROOT}/prompt_minimal_v2/return_span_logistic_v1/folds}
feature_modes=(unmasked masked fusion)
mode_index=$((SLURM_ARRAY_TASK_ID / 9))
year_index=$((SLURM_ARRAY_TASK_ID % 9))
feature_mode=${feature_modes[$mode_index]}
test_year=$((2018 + year_index))
destination=${OUTPUT_ROOT}/${feature_mode}/next_day_return/${test_year}
if [[ -s "${destination}/COMPLETED" ]]; then exit 0; fi
cd "${REPO_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
.venv/bin/python scripts/run_return_span_logistic_fold.py \
  --span-cache-root "${SPAN_ROOT}/cache" \
  --mechanism-cache-root "${MECHANISM_ROOT}/cache" \
  --output-root "${OUTPUT_ROOT}" \
  --feature-mode "${feature_mode}" --target next_day_return --test-year "${test_year}"
