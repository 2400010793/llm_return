#!/bin/bash
#SBATCH --job-name=sina-stock-hier
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/sina_semantic_span/%A_%a_hier.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/sina_semantic_span/%A_%a_hier.err
set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
MANIFEST=${MANIFEST:-${REPO_ROOT}/configs/generated/sina_stock_hierarchy_prototype.tsv}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/semantic_span_v1}
cd "${REPO_ROOT}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id model prompt_length semantic_name variant target cluster_method predictor <<< "${line%$'\r'}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || exit 2
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
.venv/bin/python scripts/run_prompt_span_return_regression.py \
  --span-cache-root "${STUDY_ROOT}/cache" \
  --mechanism-cache-root "${SINA_ROOT}/prompt_minimal_v2/analysis_v1/cache" \
  --output-root "${STUDY_ROOT}/stock_hierarchy_prototype" \
  --model "${model}" --prompt-length "${prompt_length}" \
  --semantic-name "${semantic_name}" --variant "${variant}" \
  --target "${target}" --components 32 --cluster-augmented \
  --cluster-method agglomerative --hierarchy-sample 4000 --predictor ridge
