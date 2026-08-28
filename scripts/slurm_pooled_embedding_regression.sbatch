#!/bin/bash
#SBATCH --job-name=pooled-regression
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_pooled_regression/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_pooled_regression/%A_%a.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?regression array must be submitted through scripts/task_tracker.py}"
: "${MANIFEST:?export MANIFEST}"
: "${SLURM_ARRAY_TASK_ID:?run as an array}"
PANEL=${PANEL:-data/processed/cninfo_full_o2o_panel.parquet}
SEARCH_STAGE=${SEARCH_STAGE:-coarse}
ALPHAS=${ALPHAS:-1e-5,1e-4,1e-3,1e-2,1e-1,1,10,50,100}
if [[ -n "${ALPHAS_ENCODED:-}" ]]; then
  ALPHAS=${ALPHAS_ENCODED//;/,}
fi
MAX_ALPHA_EXPANSIONS=${MAX_ALPHA_EXPANSIONS:-0}
SELECTION_CORRELATION=${SELECTION_CORRELATION:-spearman}
EXPECTED_EMBEDDING_ROWS=${EXPECTED_EMBEDDING_ROWS:-350577}
FIT_WINDOW_YEARS=${FIT_WINDOW_YEARS:-6}
VALIDATION_WINDOW_YEARS=${VALIDATION_WINDOW_YEARS:-2}
mkdir -p logs/slurm_pooled_regression reports/regression/pooled_embeddings/stock_day
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id embedding_root model variant feature target regressor run_mode reducer components output <<< "${line}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || { echo "manifest task mismatch" >&2; exit 2; }

/usr/bin/python3 scripts/run_pooled_embedding_regression.py \
  "${PANEL}" \
  --embedding-root "${embedding_root}" \
  --model "${model}" --variant "${variant}" --feature "${feature}" \
  --target-column "${target}" \
  --regressor "${regressor}" --run-mode "${run_mode}" \
  --reducer "${reducer}" --reducer-components "${components:-128}" \
  --alphas "${ALPHAS}" --max-alpha-expansions "${MAX_ALPHA_EXPANSIONS}" \
  --search-stage "${SEARCH_STAGE}" \
  --selection-correlation "${SELECTION_CORRELATION}" \
  --fit-window-years "${FIT_WINDOW_YEARS}" \
  --validation-window-years "${VALIDATION_WINDOW_YEARS}" \
  --expected-embedding-rows "${EXPECTED_EMBEDDING_ROWS}" \
  --seed 42 --output "${output}"
