#!/bin/bash
#SBATCH --job-name=long-pool-full
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-63%8
#SBATCH --output=/home/team/llm_return/logs/slurm_long_pool_full/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_long_pool_full/%A_%a.err
set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_long_pool_full data/processed/pooled_long_embeddings_v4
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
shard=$((SLURM_ARRAY_TASK_ID / 2))
model_id=$((SLURM_ARRAY_TASK_ID % 2))
case "$model_id" in
  0) model=roberta; max_length=512; model_name=roberta ;;
  1) model=bge_m3; max_length=1000; model_name=bge_m3 ;;
esac
input="data/processed/cleaned/prompt_v4_fixed_long_parts/shard-${shard}"
base="data/processed/pooled_long_embeddings_v4/shard-${shard}/${model_name}"
mkdir -p "$base"
for variant in long masked_long; do
  /usr/bin/python3 scripts/run_short_pooled_embeddings.py "$input" \
    --output-dir "$base/$variant" --model "$model" --variant "$variant" \
    --batch-size 4 --max-length "$max_length"
done
