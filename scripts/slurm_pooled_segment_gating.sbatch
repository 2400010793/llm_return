#!/bin/bash
#SBATCH --job-name=segment-gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_segment_gate/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_segment_gate/%A_%a.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${MANIFEST:?export MANIFEST to the segment-gating TSV manifest}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"
PANEL=${PANEL:-data/processed/cninfo_full_classification_panel.parquet}
mkdir -p logs/slurm_segment_gate reports/classification/pooled_embeddings/segment_gating
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
IFS=$'\t' read -r task_id embedding_root model variant feature classifier gate_method gate_keep output train_target evaluation_target <<< "${line}"
if [[ "${task_id}" != "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "Manifest task mismatch: expected ${SLURM_ARRAY_TASK_ID}, found ${task_id}" >&2
  exit 2
fi
: "${train_target:?manifest must provide train_target}"
: "${evaluation_target:?manifest must provide evaluation_target}"

/usr/bin/python3 scripts/run_pooled_embedding_classification.py \
  "${PANEL}" \
  --embedding-root "${embedding_root}" \
  --model "${model}" \
  --variant "${variant}" \
  --feature "${feature}" \
  --classifier "${classifier}" \
  --reducer group_gate \
  --group-gate-method "${gate_method}" \
  --group-gate-keep "${gate_keep}" \
  --scaler auto \
  --train-target-column "${train_target}" \
  --evaluation-target-column "${evaluation_target}" \
  --date-column entry_date \
  --panel-row-index-column row_index \
  --search-stage coarse \
  --seed 42 \
  --output "${output}"
