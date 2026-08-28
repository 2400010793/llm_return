#!/bin/bash
#SBATCH --job-name=return-span-cluster
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/return_span_cluster/%A_%a_fold.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/return_span_cluster/%A_%a_fold.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?run as array}"
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
MECHANISM_ROOT=${MECHANISM_ROOT:-${SINA_ROOT}/prompt_minimal_v2/analysis_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/return_span_cluster_v1}
MANIFEST=${FOLD_MANIFEST:-${REPO_ROOT}/configs/generated/minimal_prompt_knn_folds_v2.tsv}
cd "${REPO_ROOT}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id model prompt_length variant target test_year <<< "${line%$'\r'}"
destination=${STUDY_ROOT}/folds/${model}/${prompt_length}/${variant}/${target}/${test_year}
if [[ -s "${destination}/COMPLETED" ]]; then exit 0; fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
.venv/bin/python scripts/run_prompt_return_span_cluster_fold.py \
  --span-cache-root "${STUDY_ROOT}/cache" \
  --mechanism-cache-root "${MECHANISM_ROOT}/cache" \
  --output-root "${STUDY_ROOT}/folds" --model "${model}" \
  --prompt-length "${prompt_length}" --variant "${variant}" \
  --target "${target}" --test-year "${test_year}"
