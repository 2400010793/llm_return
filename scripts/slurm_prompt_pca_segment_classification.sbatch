#!/bin/bash
#SBATCH --job-name=prompt-pca-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_prompt_pca_fusion_cls/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_prompt_pca_fusion_cls/%A_%a.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${MANIFEST:?export the experiment manifest}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"

PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
EXPECTED_ROWS=${EXPECTED_ROWS:-903665}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id variant feature matrix metadata classifier output <<< "${line}"
if [[ -z "${line}" || "${task_id}" != "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "Manifest task mismatch for ${SLURM_ARRAY_TASK_ID}: ${line}" >&2
  exit 2
fi

mkdir -p logs/slurm_prompt_pca_fusion_cls "$(dirname "${output}")"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

/usr/bin/python3 scripts/run_precomputed_embedding_classification.py "${PANEL}" \
  --matrix "${matrix}" \
  --metadata "${metadata}" \
  --embedding-model chinese_roberta \
  --input-name "${variant}_${feature}" \
  --classifier "${classifier}" \
  --reducer none \
  --scaler auto \
  --metadata-row-index-offset 0 \
  --expected-rows "${EXPECTED_ROWS}" \
  --train-target-column next_day_return \
  --evaluation-target-column next_day_return \
  --date-column entry_date \
  --panel-row-index-column row_index \
  --search-stage coarse \
  --seed 42 \
  --output "${output}"
