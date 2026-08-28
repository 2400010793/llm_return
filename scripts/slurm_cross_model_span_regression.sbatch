#!/bin/bash
#SBATCH --job-name=xmodel-span
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/cross_model_span/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/cross_model_span/%A_%a.err
set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?array task required}"
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
SPAN_ROOT=${SPAN_ROOT:-${SINA_ROOT}/prompt_minimal_v2/semantic_span_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-${SINA_ROOT}/prompt_minimal_v2/cross_model_span_v3}
MANIFEST=${MANIFEST:-${REPO_ROOT}/configs/generated/cross_model_span_manifest.tsv}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id semantics variant target components cluster mode <<< "${line%$'\r'}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || exit 2
cd "${REPO_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cluster_arg=()
if [[ "${cluster}" == "1" ]]; then cluster_arg=(--cluster); fi
.venv/bin/python scripts/run_cross_model_span_regression.py \
  --span-cache-root "${SPAN_ROOT}/cache" \
  --mechanism-cache-root "${SINA_ROOT}/prompt_minimal_v2/analysis_v1/cache" \
  --output-root "${OUTPUT_ROOT}/${semantics}/${cluster}/components_${components}" \
  --semantics "${semantics}" --variant "${variant}" --target "${target}" \
  --components "${components}" "${cluster_arg[@]}"
