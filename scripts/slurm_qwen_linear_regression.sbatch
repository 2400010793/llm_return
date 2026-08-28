#!/bin/bash
#SBATCH --job-name=qwen-linear-reg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_qwen_linear/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_qwen_linear/%A_%a.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
: "${MANIFEST:?export MANIFEST}"
: "${SLURM_ARRAY_TASK_ID:?run as an array}"
PANEL=${PANEL:-data/processed/cninfo_full_o2o_residual_panel.parquet}
FIT_WINDOW_YEARS=${FIT_WINDOW_YEARS:-6}
VALIDATION_WINDOW_YEARS=${VALIDATION_WINDOW_YEARS:-2}
SEARCH_STAGE=${SEARCH_STAGE:-fine}
mkdir -p logs/slurm_qwen_linear
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id model embedding_root variant target regressor run_mode reducer components seed output <<< "${line}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || { echo "manifest task mismatch" >&2; exit 2; }

.venv/bin/python scripts/run_pooled_embedding_regression.py "${PANEL}" \
  --embedding-root "${embedding_root}" --model "${model}" \
  --variant "${variant}" --feature full_mean --target-column "${target}" \
  --regressor "${regressor}" --run-mode "${run_mode}" \
  --reducer "${reducer}" --reducer-components "${components}" \
  --search-stage "${SEARCH_STAGE}" --fit-window-years "${FIT_WINDOW_YEARS}" \
  --validation-window-years "${VALIDATION_WINDOW_YEARS}" \
  --expected-embedding-rows 350577 --max-embedding-matrix-gib 16 \
  --seed "${seed}" --output "${output}"
