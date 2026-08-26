#!/bin/bash
#SBATCH --job-name=precomp-model
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_precomputed_models/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_precomputed_models/%A_%a.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?model array must be submitted through scripts/task_tracker.py}"
: "${MANIFEST:?export MANIFEST to a generated model-only TSV manifest}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"
PANEL=${PANEL:-data/processed/cninfo_full_classification_panel.parquet}
mkdir -p logs/slurm_precomputed_models reports/classification/precomputed_embeddings
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
IFS=$'\t' read -r task_id phase embedding_model input_name matrix metadata classifier reducer components seed output train_target evaluation_target <<< "${line}"
if [[ "${task_id}" != "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "Manifest task mismatch: expected ${SLURM_ARRAY_TASK_ID}, found ${task_id}" >&2
  exit 2
fi
mkdir -p "$(dirname "${output}")"
train_target=${train_target:-next_day_return}
evaluation_target=${evaluation_target:-next_day_return}
if [[ "${train_target}" == "event_return_3d" && "${phase}" != "label_comparison" ]]; then
  echo "Refusing legacy event3 target outside label_comparison: phase=${phase}" >&2
  exit 2
fi

/usr/bin/python3 scripts/run_precomputed_embedding_classification.py \
  "${PANEL}" \
  --matrix "${matrix}" \
  --metadata "${metadata}" \
  --embedding-model "${embedding_model}" \
  --input-name "${input_name}" \
  --classifier "${classifier}" \
  --reducer "${reducer}" \
  --reducer-components "${components:-128}" \
  --scaler auto \
  --metadata-row-index-offset 1 \
  --expected-rows 350577 \
  --train-target-column "${train_target}" \
  --evaluation-target-column "${evaluation_target}" \
  --date-column entry_date \
  --panel-row-index-column row_index \
  --search-stage coarse \
  --seed "${seed}" \
  --output "${output}"
