#!/bin/bash
#SBATCH --job-name=ss-mlp-seeds
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_simple_states_seeds/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_simple_states_seeds/%A_%a.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?seed array must use scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?run as an array job}"
: "${MANIFEST:?export MANIFEST}"
PANEL=${PANEL:-data/processed/cninfo_full_simple_states_panel_2010_2026.parquet}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-data/processed/pooled_embeddings_2010_2026_clean_v1}
mkdir -p logs/slurm_simple_states_seeds
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id seed output <<< "${line}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || {
  echo "manifest task mismatch" >&2
  exit 2
}

/usr/bin/python3 scripts/run_pooled_embedding_regression.py "${PANEL}" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --model roberta --variant short --feature full_mean \
  --target-column simple_states_ret_vv_return \
  --regressor small_mlp --run-mode final-test \
  --reducer none --reducer-components 0 \
  --search-stage fine --selection-correlation pearson \
  --expected-embedding-rows 903665 --max-embedding-matrix-gib 64 \
  --seed "${seed}" --output "${output}"
