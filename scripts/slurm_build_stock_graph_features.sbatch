#!/bin/bash
#SBATCH --job-name=stock-graph-build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_stock_graph/%j_build.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_stock_graph/%j_build.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"

PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-data/processed/pooled_embeddings_2010_2026_roberta_v1}
UNIVERSE=${UNIVERSE:-data/stock_universe_paper_1000.csv}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/processed/stock_graph_features_roberta_masked_v1}
mkdir -p logs/slurm_stock_graph
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

/usr/bin/python3 scripts/build_stock_graph_features.py "${PANEL}" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --universe "${UNIVERSE}" \
  --output-dir "${OUTPUT_ROOT}" \
  --model roberta \
  --variant masked_short \
  --feature title_body_concat \
  --components 256 \
  --fit-sample-rows 10000 \
  --expected-rows 903665 \
  --seed 42
