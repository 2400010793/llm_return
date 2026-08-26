#!/bin/bash
#SBATCH --job-name=frozen-predict-v1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --requeue
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_two_stage_v1/predict_%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_two_stage_v1/predict_%A_%a.err

set -euo pipefail
REPO_ROOT=/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${MANIFEST:?export MANIFEST}"
: "${SLURM_ARRAY_TASK_ID:?run as an array}"

mkdir -p logs/slurm_two_stage_v1 reports/frozen_gate/two_stage_v1_predictors
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id model variant token_count hidden_size window gate_tag gate_mode representation_size gate_hidden_size dropout learning_rate weight_decay inner_fit_years inner_validation_years predictor_fit_years predictor_validation_years seed representation_report predictor_output <<< "${line}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || { echo "manifest task mismatch" >&2; exit 2; }

/usr/bin/python3 scripts/run_frozen_representation_predictors.py \
  data/processed/cninfo_full_classification_panel.parquet \
  --representation-report "${representation_report}" --pca-components 16,32 \
  --seed "${seed}" --output "${predictor_output}"
