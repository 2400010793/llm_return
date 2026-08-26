#!/bin/bash
#SBATCH --job-name=soft-family-v2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/soft_return_family/%A_%a_v2.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/soft_return_family/%A_%a_v2.err
set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
MANIFEST=${MANIFEST:-${REPO_ROOT}/configs/generated/soft_return_family_v2.tsv}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/soft_return_family_v2}
cd "${REPO_ROOT}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id model feature_mode target test_year <<< "${line%$'\r'}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || exit 2
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
.venv/bin/python scripts/run_soft_return_family_fold.py \
  --span-cache-root "${SINA_ROOT}/prompt_minimal_v2/semantic_span_v1/cache" \
  --mechanism-cache-root "${SINA_ROOT}/prompt_minimal_v2/analysis_v1/cache" \
  --output-root "${STUDY_ROOT}/folds/${model}" --model "${model}" \
  --feature-mode "${feature_mode}" --target "${target}" --test-year "${test_year}" \
  --one-way-cost-bps 5
