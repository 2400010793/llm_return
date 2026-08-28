#!/bin/bash
#SBATCH --job-name=tb-seed-eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_title_body_seeds/%j_eval.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_title_body_seeds/%j_eval.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?evaluation must be submitted through scripts/task_tracker.py}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

/usr/bin/python3 scripts/evaluate_partial_title_body_seed_stability.py \
  --panel data/processed/cninfo_full_classification_panel_2010_2026.parquet \
  --embedding-root data/processed/pooled_embeddings_2010_2026_roberta_v1 \
  --artifact-root reports/classification/pooled_embeddings/title_body_selection_metric_v1 \
  --test-year 2018 \
  --output reports/classification/pooled_embeddings/title_body_selection_metric_v1/partial_2018_seed_stability.json
