#!/bin/bash
# Frozen v5 PCA-128: 2 models x 4 variants x (Random Forest, exact KNN).
# NB variants are excluded because PCA features are signed dense coordinates.
#SBATCH --job-name=prompt-pca-extra
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --array=0-15%2
#SBATCH --output=/home/team/llm_return/logs/slurm_prompt_pca_extra_v5/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_prompt_pca_extra_v5/%A_%a.err
set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_prompt_pca_extra_v5 reports/classification/prompt_tokens_v5
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
models=(roberta bge_m3)
variants=(short masked_short long masked_long)
classifiers=(random_forest knn)
model=${models[$((SLURM_ARRAY_TASK_ID / 8))]}
variant=${variants[$(((SLURM_ARRAY_TASK_ID / 2) % 4))]}
classifier=${classifiers[$((SLURM_ARRAY_TASK_ID % 2))]}
matrix="data/processed/prompt_token_pca_v5/${model}/${variant}/pca_128.npy"
output="reports/classification/prompt_tokens_v5/${model}_${variant}_${classifier}_pca128.json"
/usr/bin/python3 scripts/run_reduced_prompt_classification.py \
  data/processed/cninfo_full_classification_panel.parquet \
  --matrix "${matrix}" --model "${model}" --variant "${variant}" \
  --classifier "${classifier}" --components 128 --seed 42 --output "${output}"