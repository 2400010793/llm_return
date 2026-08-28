#!/bin/bash
#SBATCH --job-name=overnight-pooled-v1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/home/team/llm_return/logs/slurm_overnight_v1/pooled_%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_overnight_v1/pooled_%A_%a.err

set -euo pipefail
REPO_ROOT=/home/team/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${MANIFEST:?export MANIFEST}"
: "${SLURM_ARRAY_TASK_ID:?run as an array}"

mkdir -p logs/slurm_overnight_v1 reports/regression/overnight_v1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id family embedding_root model variant feature target regressor run_mode reducer components search_stage alphas output <<< "${line}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || { echo "manifest task mismatch" >&2; exit 2; }

/usr/bin/python3 scripts/run_pooled_embedding_regression.py \
  data/processed/cninfo_full_classification_panel.parquet \
  --embedding-root "${embedding_root}" --model "${model}" --variant "${variant}" \
  --feature "${feature}" --target-column "${target}" --regressor "${regressor}" \
  --run-mode "${run_mode}" --reducer "${reducer}" --reducer-components "${components:-128}" \
  --search-stage "${search_stage}" --alphas "${alphas}" \
  --expected-embedding-rows 350577 --seed 42 --output "${output}"
