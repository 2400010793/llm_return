#!/bin/bash
#SBATCH --job-name=soft-return-family
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/soft_return_family/%A_%a_fold.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/soft_return_family/%A_%a_fold.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?run as array}"
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
MECHANISM_ROOT=${MECHANISM_ROOT:-${SINA_ROOT}/prompt_minimal_v2/analysis_v1}
SPAN_ROOT=${SPAN_ROOT:-${SINA_ROOT}/prompt_minimal_v2/return_span_cluster_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/soft_return_family_v1}
MANIFEST=${FAMILY_MANIFEST:-${REPO_ROOT}/configs/generated/soft_return_family_v1.tsv}
cd "${REPO_ROOT}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id feature_mode target test_year <<< "${line%$'\r'}"
destination=${STUDY_ROOT}/folds/${feature_mode}/${target}/${test_year}
if [[ -s "${destination}/COMPLETED" ]]; then exit 0; fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
.venv/bin/python scripts/run_soft_return_family_fold.py \
  --span-cache-root "${SPAN_ROOT}/cache" --mechanism-cache-root "${MECHANISM_ROOT}/cache" \
  --output-root "${STUDY_ROOT}/folds" --feature-mode "${feature_mode}" \
  --target "${target}" --test-year "${test_year}" --one-way-cost-bps 5
