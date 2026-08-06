#!/bin/bash
#SBATCH --job-name=zh-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-5%3
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_zh_classification/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_zh_classification/%A_%a.err

set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_zh_classification reports/classification/chinese_transformers
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

models=(roberta bge_m3)
files=(data/processed/embeddings/chinese_roberta/chinese_roberta.npy data/processed/embeddings/bge_m3/bge_m3.npy)
model=${models[$((SLURM_ARRAY_TASK_ID / 3))]}
embedding=${files[$((SLURM_ARRAY_TASK_ID / 3))]}
classifiers=(logistic random_forest mlp)
classifier=${classifiers[$((SLURM_ARRAY_TASK_ID % 3))]}
output="reports/classification/chinese_transformers/${model}_${classifier}_seed42.json"

/usr/bin/python3 scripts/run_classification.py \
  data/processed/cninfo_100_classification_panel.parquet \
  --train-end 2024-01-01 \
  --predict-start 2024-01-01 \
  --representations "" \
  --${model}-npy "${embedding}" \
  --classifiers "${classifier}" \
  --seed 42 \
  --tune \
  --search-stage coarse \
  --validation-fraction 0.2 \
  --max-input-chars 12000 \
  --output "${output}"
