#!/bin/bash
#SBATCH --job-name=finbert2-stable-mlp
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_finbert2_stable_mlp/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_finbert2_stable_mlp/%A_%a.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?array must be submitted through scripts/task_tracker.py}"
: "${MANIFEST:?export MANIFEST}"
: "${SLURM_ARRAY_TASK_ID:?run as a Slurm array}"

PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-data/processed/pooled_finbert2_embeddings_2010_2026_v1}
EXPECTED_ROWS=${EXPECTED_ROWS:-903665}

mkdir -p logs/slurm_finbert2_stable_mlp
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id model variant feature seed schedule min_epoch output <<< "${line}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || {
  echo "manifest task mismatch: expected ${SLURM_ARRAY_TASK_ID}, found ${task_id}" >&2
  exit 2
}

mkdir -p "$(dirname "${output}")"
/usr/bin/python3 scripts/run_pooled_embedding_classification.py \
  "${PANEL}" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --model "${model}" \
  --variant "${variant}" \
  --feature "${feature}" \
  --classifier simple_mlp \
  --reducer none \
  --scaler auto \
  --train-target-column next_day_return \
  --evaluation-target-column next_day_return \
  --date-column entry_date \
  --panel-row-index-column row_index \
  --search-stage stable \
  --learning-rate-schedule "${schedule}" \
  --early-stopping-metric accuracy \
  --early-stopping-min-epoch "${min_epoch}" \
  --final-epoch-multiplier 1.0 \
  --seed "${seed}" \
  --expected-embedding-rows "${EXPECTED_ROWS}" \
  --output "${output}"
