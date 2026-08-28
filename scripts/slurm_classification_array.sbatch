#!/bin/bash
#SBATCH --job-name=cninfo-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-11%4
#SBATCH --output=/home/team/llm_return/logs/slurm_classification/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_classification/%A_%a.err

set -euo pipefail
cd /home/team/llm_return
mkdir -p reports/classification/slurm_coarse logs/slurm_classification
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

representations=(word_tfidf char_tfidf lexicon word2vec)
classifiers=(logistic random_forest mlp)
representation=${representations[$((SLURM_ARRAY_TASK_ID / 3))]}
classifier=${classifiers[$((SLURM_ARRAY_TASK_ID % 3))]}
output="reports/classification/slurm_coarse/${representation}_${classifier}_seed42.json"

/usr/bin/python3 scripts/run_classification.py \
  data/processed/cninfo_100_classification_panel.parquet \
  --train-end 2024-01-01 \
  --predict-start 2024-01-01 \
  --representations "${representation}" \
  --classifiers "${classifier}" \
  --seed 42 \
  --tune \
  --search-stage coarse \
  --validation-fraction 0.2 \
  --max-input-chars 12000 \
  --max-features 50000 \
  --word2vec-size 100 \
  --word2vec-epochs 10 \
  --output "${output}"
