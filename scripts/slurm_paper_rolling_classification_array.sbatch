#!/bin/bash
#SBATCH --job-name=paper-roll-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-11%4
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_paper_rolling/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_paper_rolling/%A_%a.err

set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p reports/classification/paper_rolling logs/slurm_paper_rolling
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

representations=(word_tfidf char_tfidf lexicon word2vec)
classifiers=(logistic random_forest mlp)
representation=${representations[$((SLURM_ARRAY_TASK_ID / 3))]}
classifier=${classifiers[$((SLURM_ARRAY_TASK_ID % 3))]}
output="reports/classification/paper_rolling/${representation}_${classifier}_return1_seed42.json"

/usr/bin/python3 scripts/run_paper_rolling_classification.py \
  data/processed/cninfo_100_classification_panel.parquet \
  --target-column return_1 \
  --representations "${representation}" \
  --classifiers "${classifier}" \
  --seed 42 \
  --search-stage coarse \
  --max-input-chars 12000 \
  --max-features 50000 \
  --word2vec-size 100 \
  --word2vec-epochs 10 \
  --output "${output}"
