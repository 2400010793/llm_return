#!/bin/bash
#SBATCH --job-name=existing-embed-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-35%36
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_existing_embed_cls/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_existing_embed_cls/%A_%a.err
set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_existing_embed_cls reports/classification/existing_embeddings
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
models=(chinese_roberta bge_m3 word2vec)
inputs=(text_plain text_input_2_short text_input_3_long text_input_5_masked_short text_input_6_masked_long)
model=${models[$((SLURM_ARRAY_TASK_ID / 15))]}
input=${inputs[$(((SLURM_ARRAY_TASK_ID / 3) % 5))]}
classifier_index=$((SLURM_ARRAY_TASK_ID % 3))
classifiers=(logistic random_forest mlp); classifier=${classifiers[$classifier_index]}
name="${model}_${input}"
emb="data/processed/classification_embeddings/${name}.npy"
panel="data/processed/classification_embeddings/${name}.parquet"
out="reports/classification/existing_embeddings/${name}_${classifier}_seed42.json"
/usr/bin/python3 scripts/run_classification.py "$panel" \
 --train-end 2024-01-01 --predict-start 2024-01-01 --representations "" \
 --qwen-npy "$emb" --classifiers "$classifier" --seed 42 --tune --search-stage coarse \
 --validation-fraction 0.2 --max-input-chars 0 --output "$out"
