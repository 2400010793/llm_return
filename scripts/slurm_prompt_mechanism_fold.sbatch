#!/bin/bash
#SBATCH --job-name=prompt-fold
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/prompt_mechanism/%A_%a_fold.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/prompt_mechanism/%A_%a_fold.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?run as an array}"
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_mechanism_v1}
MANIFEST=${FOLD_MANIFEST:-${REPO_ROOT}/configs/generated/prompt_mechanism_folds_v1.tsv}
cd "${REPO_ROOT}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id model prompt_length variant target test_year <<< "${line%$'\r'}"
: "${model:?}" "${prompt_length:?}" "${variant:?}" "${target:?}" "${test_year:?}"
destination=${STUDY_ROOT}/results/${model}/${prompt_length}/${variant}/${target}/test_${test_year}
if [[ -s "${destination}/COMPLETED" ]]; then
  echo "reusing completed fold ${destination}"
  exit 0
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
.venv/bin/python scripts/run_prompt_mechanism_fold.py \
  --cache-root "${STUDY_ROOT}/cache" --baseline-root "${STUDY_ROOT}/baselines" \
  --output-root "${STUDY_ROOT}/results" --model "${model}" \
  --prompt-length "${prompt_length}" --variant "${variant}" \
  --target "${target}" --test-year "${test_year}" --bootstrap 500 --with-mlp
