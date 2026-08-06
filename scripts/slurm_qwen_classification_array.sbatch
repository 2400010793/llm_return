#!/bin/bash
#SBATCH --job-name=qwen-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-2%3
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_qwen_classification/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_qwen_classification/%A_%a.err

set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_qwen_classification reports/classification/qwen3_embedding_8b
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

classifiers=(logistic random_forest mlp)
classifier=${classifiers[$SLURM_ARRAY_TASK_ID]}
output="reports/classification/qwen3_embedding_8b/${classifier}_seed42.json"

/usr/bin/python3 scripts/run_classification.py \
  data/processed/cninfo_100_classification_panel.parquet \
  --train-end 2024-01-01 \
  --predict-start 2024-01-01 \
  --representations "" \
  --qwen-npy data/processed/embeddings/qwen3_embedding_8b/embeddings.npy \
  --classifiers "${classifier}" \
  --seed 42 \
  --tune \
  --search-stage coarse \
  --validation-fraction 0.2 \
  --max-input-chars 12000 \
  --output "${output}"
