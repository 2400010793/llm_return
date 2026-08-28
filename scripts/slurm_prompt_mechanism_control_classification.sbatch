#!/bin/bash
#SBATCH --job-name=prompt-control-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/prompt_mechanism/%A_%a_control_cls.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/prompt_mechanism/%A_%a_control_cls.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?run as array}"
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_mechanism_v1}
MANIFEST=${CONTROL_CLASS_MANIFEST:-${STUDY_ROOT}/controls/classification_manifest.tsv}
SELECTION=${CONTROL_SELECTION:-${STUDY_ROOT}/controls/selection.json}
cd "${REPO_ROOT}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id spec_index model source_variant control_id output_variant target <<< "${line%$'\r'}"
destination=${STUDY_ROOT}/controls/results/${model}/${output_variant}/${target}
if [[ -s "${destination}/COMPLETED" ]]; then echo "reusing ${destination}"; exit 0; fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
.venv/bin/python scripts/run_prompt_control_classification.py \
  --panel "${SINA_ROOT}/classification/sina_single_stock_classification_panel.parquet" \
  --source-embedding-root "${SINA_ROOT}/embeddings" \
  --control-embedding-root "${STUDY_ROOT}/controls/embeddings" \
  --control-selection "${SELECTION}" --spec-index "${spec_index}" \
  --model "${model}" --source-variant "${source_variant}" \
  --output-variant "${output_variant}" --target "${target}" \
  --expected-rows 75894 --bootstrap 500 --output "${destination}"
