#!/bin/bash
# Training-only hard token-position gates followed by sampled PCA.
# 2 models x 4 prompt variants x 2 gate methods. Submit only through task_tracker.py.
#SBATCH --job-name=prompt-gate-pca
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --array=0-15%2
#SBATCH --output=/home/team/llm_return/logs/slurm_prompt_gate_pca_v5/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_prompt_gate_pca_v5/%A_%a.err
set -euo pipefail
cd /home/team/llm_return
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
mkdir -p logs/slurm_prompt_gate_pca_v5 data/processed/prompt_token_pca_v5
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
models=(roberta bge_m3)
variants=(short masked_short long masked_long)
gates=(fisher variance)
model=${models[$((SLURM_ARRAY_TASK_ID / 8))]}
variant=${variants[$(((SLURM_ARRAY_TASK_ID / 2) % 4))]}
gate=${gates[$((SLURM_ARRAY_TASK_ID % 2))]}
output_dir="data/processed/prompt_token_pca_v5/${model}/${variant}"
/usr/bin/python3 scripts/run_prompt_token_pca_v5.py \
  data/processed/cninfo_full_classification_panel.parquet \
  --input-root data/processed/prompt_token_embeddings_v5 \
  --output-dir "${output_dir}" --model "${model}" --variant "${variant}" \
  --components 128 --fit-sample-rows 10000 --transform-batch-size 256 \
  --token-gate-method "${gate}" --token-gate-keep 8 --seed 42
