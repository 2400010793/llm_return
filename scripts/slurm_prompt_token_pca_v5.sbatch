#!/bin/bash
# 2 models x 4 variants; fit sampled initial-training PCA once per combination.
#SBATCH --job-name=prompt-pca-v5
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --array=0-7%2
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_prompt_pca_v5/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_prompt_pca_v5/%A_%a.err
set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_prompt_pca_v5 data/processed/prompt_token_pca_v5
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
models=(roberta bge_m3)
variants=(short masked_short long masked_long)
model=${models[$((SLURM_ARRAY_TASK_ID / 4))]}
variant=${variants[$((SLURM_ARRAY_TASK_ID % 4))]}
/usr/bin/python3 scripts/run_prompt_token_pca_v5.py \
  data/processed/cninfo_full_classification_panel.parquet \
  --model "${model}" --variant "${variant}" \
  --components 128 --fit-sample-rows 10000 --transform-batch-size 256 \
  --output-dir "data/processed/prompt_token_pca_v5/${model}/${variant}"