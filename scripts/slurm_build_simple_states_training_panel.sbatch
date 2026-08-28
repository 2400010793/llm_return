#!/bin/bash
#SBATCH --job-name=ss-train-panel
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_simple_states/%j.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_simple_states/%j.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?training-panel build must use scripts/task_tracker.py}"
SOURCE_PANEL=${SOURCE_PANEL:-data/processed/cninfo_full_o2o_panel_2010_2026_hfq.parquet}
OUTPUT_PANEL=${OUTPUT_PANEL:-data/processed/cninfo_full_simple_states_panel_2010_2026.parquet}
DATA_ROOT=${DATA_ROOT:-/data/alpha_team2/shares/simple_states_data}
mkdir -p logs/slurm_simple_states "$(dirname "${OUTPUT_PANEL}")"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

/usr/bin/python3 scripts/build_simple_states_training_panel.py "${SOURCE_PANEL}" \
  --output "${OUTPUT_PANEL}" --data-root "${DATA_ROOT}" \
  --min-amount 20000000 --amount-window 20
