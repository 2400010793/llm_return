#!/bin/bash
#SBATCH --job-name=qwen-reg-screen
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/home/team/llm_return/logs/slurm_qwen_overnight_20260811/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_qwen_overnight_20260811/%A_%a.err

set -euo pipefail
REPO_ROOT=/home/team/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${MANIFEST:?export the regression manifest}"
: "${SLURM_ARRAY_TASK_ID:?run as an array job}"
mkdir -p logs/slurm_qwen_overnight_20260811 \
  reports/regression/qwen_fullpanel_screen_20260811
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id variant target regressor reducer components output <<< "${line}"
if [[ -z "${line}" || "${task_id}" != "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "Manifest mismatch for array task ${SLURM_ARRAY_TASK_ID}: ${line}" >&2
  exit 2
fi

"${REPO_ROOT}/.venv/bin/python" scripts/run_pooled_embedding_regression.py \
  data/processed/cninfo_full_o2o_residual_panel.parquet \
  --embedding-root data/processed/pooled_embeddings_qwen_candidates_1000 \
  --model qwen3_embedding_8b --variant "${variant}" --feature full_mean \
  --target-column "${target}" --regressor "${regressor}" \
  --reducer "${reducer}" --reducer-components "${components}" \
  --run-mode screen --search-stage coarse --seed 42 \
  --expected-embedding-rows 350577 --max-embedding-matrix-gib 8 \
  --output "${output}"
