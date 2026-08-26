#!/bin/bash
#SBATCH --job-name=min-prompt-sum
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/minimal_prompt_knn/%j_summary.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/minimal_prompt_knn/%j_summary.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/analysis_v1}
EXPECTED_FOLDS=${EXPECTED_FOLDS:-144}
SUBMIT_SIMPLE_STATES=${SUBMIT_SIMPLE_STATES:-0}
SUMMARY_SUBDIR=${SUMMARY_SUBDIR:-summary}
MODELS=${MODELS:-}
cd "${REPO_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
model_args=()
for model in ${MODELS}; do model_args+=(--model "${model}"); done
.venv/bin/python scripts/summarize_minimal_prompt_knn_v2.py \
  --fold-root "${STUDY_ROOT}/folds" --output-root "${STUDY_ROOT}/${SUMMARY_SUBDIR}" \
  --expected-folds "${EXPECTED_FOLDS}" "${model_args[@]}" --strict
if [[ "${SUBMIT_SIMPLE_STATES}" == "1" ]]; then
  env -u TASK_RECORD_ID \
    MANIFEST="${STUDY_ROOT}/${SUMMARY_SUBDIR}/simple_states_manifest.tsv" \
    OUTPUT_ROOT="${STUDY_ROOT}/${SUMMARY_SUBDIR}/simple_states/results" \
    TASK_NAME=cninfo-minimal-short-simple-states-v2 \
    PARTITION="${SLURM_JOB_PARTITION:-cpu}" \
    MAX_PARALLEL=16 TOP_N=100 MIN_COUNT=20 MIN_AMOUNT=20000000 \
    bash scripts/submit_simple_states_evaluation.sh
fi
