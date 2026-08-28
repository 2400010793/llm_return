#!/bin/bash
#SBATCH --job-name=o2o-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_o2o_classification/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_o2o_classification/%A_%a.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?model array must be submitted through scripts/task_tracker.py}"
: "${MANIFEST:?export MANIFEST to a generated TSV manifest}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"
PANEL=${PANEL:-data/processed/cninfo_full_o2o_panel.parquet}
FILTER_COLUMN=${FILTER_COLUMN:-o2o_target_eligible}
mkdir -p logs/slurm_o2o_classification reports/classification/pooled_embeddings/o2o_core
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
if [[ -z "${line}" ]]; then
  echo "No task ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
  exit 2
fi
IFS=$'\t' read -r task_id phase embedding_root model variant feature classifier reducer components search_stage seed output train_target evaluation_target <<< "${line}"
if [[ "${task_id}" != "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "Manifest task mismatch: expected ${SLURM_ARRAY_TASK_ID}, found ${task_id}" >&2
  exit 2
fi
if [[ "${train_target}" != "next_day_open_to_open_return" || "${evaluation_target}" != "next_day_open_to_open_return" ]]; then
  echo "Refusing non-O2O targets: train=${train_target} evaluation=${evaluation_target}" >&2
  exit 2
fi
mkdir -p "$(dirname "${output}")"

/usr/bin/python3 scripts/run_pooled_embedding_classification.py \
  "${PANEL}" \
  --embedding-root "${embedding_root}" \
  --model "${model}" \
  --variant "${variant}" \
  --feature "${feature}" \
  --classifier "${classifier}" \
  --reducer "${reducer}" \
  --reducer-components "${components:-128}" \
  --scaler auto \
  --train-target-column "${train_target}" \
  --evaluation-target-column "${evaluation_target}" \
  --date-column entry_date \
  --panel-row-index-column row_index \
  --filter-column "${FILTER_COLUMN}" \
  --search-stage "${search_stage}" \
  --seed "${seed}" \
  --output "${output}"
