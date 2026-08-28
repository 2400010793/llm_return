#!/bin/bash
# 2 models x 4 variants x 4 classifiers on frozen training-only PCA features.
#SBATCH --job-name=prompt-pca-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-31%8
#SBATCH --output=/home/team/llm_return/logs/slurm_prompt_pca_cls_v5/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_prompt_pca_cls_v5/%A_%a.err
set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_prompt_pca_cls_v5 reports/classification/prompt_tokens_v5
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
models=(roberta bge_m3)
variants=(short masked_short long masked_long)
classifiers=(logistic linear_svm sgd mlp)
model=${models[$((SLURM_ARRAY_TASK_ID / 16))]}
variant=${variants[$(((SLURM_ARRAY_TASK_ID / 4) % 4))]}
classifier=${classifiers[$((SLURM_ARRAY_TASK_ID % 4))]}
matrix="data/processed/prompt_token_pca_v5/${model}/${variant}/pca_128.npy"
output="reports/classification/prompt_tokens_v5/${model}_${variant}_${classifier}_pca128.json"
/usr/bin/python3 scripts/run_reduced_prompt_classification.py \
  data/processed/cninfo_full_classification_panel.parquet \
  --matrix "${matrix}" --model "${model}" --variant "${variant}" \
  --classifier "${classifier}" --components 128 --seed 42 --output "${output}"