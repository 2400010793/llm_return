#!/bin/bash
# Build the row-aligned return-label panel used by pooled embedding classifiers.
#SBATCH --job-name=cninfo-full-panel
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/home/team/llm_return/logs/slurm_full_panel/%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_full_panel/%j.err

set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_full_panel
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

/usr/bin/python3 -m py_compile \
  scripts/build_cninfo_full_classification_panel.py \
  scripts/run_pooled_embedding_classification.py

/usr/bin/python3 scripts/build_cninfo_full_classification_panel.py \
  --input data/processed/cleaned/cninfo_announcements_final.jsonl \
  --returns /home/team/ret.parquet \
  --output data/processed/cninfo_full_classification_panel.parquet