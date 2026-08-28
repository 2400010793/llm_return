#!/bin/bash
#SBATCH --job-name=min-prompt-knn
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/minimal_prompt_knn/%A_%a_fold.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/minimal_prompt_knn/%A_%a_fold.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?run as an array}"
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/analysis_v1}
CACHE_ROOT=${CACHE_ROOT:-${STUDY_ROOT}/cache}
MANIFEST=${FOLD_MANIFEST:-${REPO_ROOT}/configs/generated/minimal_prompt_knn_folds_v2.tsv}
cd "${REPO_ROOT}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id model prompt_length variant target test_year <<< "${line%$'\r'}"
: "${model:?missing model}" "${variant:?missing variant}" "${target:?missing target}" "${test_year:?missing year}"
destination=${STUDY_ROOT}/folds/${model}/${prompt_length}/${variant}/${target}/${test_year}
if [[ -s "${destination}/COMPLETED" ]]; then
  echo "reusing ${destination}"
  exit 0
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
span_args=()
if [[ -n "${SPAN_CACHE_ROOT:-}" ]]; then
  span_args+=(--span-cache-root "${SPAN_CACHE_ROOT}")
fi
metadata_args=()
if [[ -n "${METADATA_MODEL:-}" ]]; then
  metadata_args+=(--metadata-model "${METADATA_MODEL}")
fi
.venv/bin/python scripts/run_minimal_prompt_knn_fold.py \
  --cache-root "${CACHE_ROOT}" --baseline-root "${STUDY_ROOT}/prompt_only" \
  --output-root "${STUDY_ROOT}/folds" --model "${model}" \
  --prompt-length "${prompt_length}" --variant "${variant}" \
  --target "${target}" --test-year "${test_year}" --jobs "${SLURM_CPUS_PER_TASK:-8}" \
  "${span_args[@]}" "${metadata_args[@]}"
