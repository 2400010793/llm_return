#!/bin/bash
#SBATCH --job-name=next1-dm-mcs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_next1_comparison/%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_next1_comparison/%j.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?comparison must be submitted through scripts/task_tracker.py}"
: "${COMPARISON_MANIFEST:?export COMPARISON_MANIFEST to model/path TSV}"
PANEL=${PANEL:-data/processed/cninfo_full_classification_panel.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-reports/forecast_comparison/classification_next1_strong}
ROW_COLUMN=${ROW_COLUMN:-row_index}
mkdir -p logs/slurm_next1_comparison "${OUTPUT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

arguments=()
while IFS=$'\t' read -r model path; do
  if [[ "${model}" == "model" ]]; then
    continue
  fi
  if [[ ! -f "${path}" ]]; then
    echo "Missing completed prediction file: ${path}" >&2
    exit 2
  fi
  arguments+=(--prediction "${model}=${path}")
done < "${COMPARISON_MANIFEST}"

/usr/bin/python3 scripts/run_forecast_comparison.py \
  --task classification \
  --panel "${PANEL}" \
  --row-column "${ROW_COLUMN}" \
  "${arguments[@]}" \
  --dm-lag 5 \
  --horizon 1 \
  --mcs-alpha 0.10 \
  --mcs-reps 10000 \
  --block-lengths 5,10,20 \
  --seed 42 \
  --output-dir "${OUTPUT_DIR}"
