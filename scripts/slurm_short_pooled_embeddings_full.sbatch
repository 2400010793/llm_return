#!/bin/bash
# Full pooled embedding run: 32 data shards x 2 models, with both short variants.
# At most 16 tasks run concurrently. Each task writes one shard/model and
# processes raw and masked short prompts separately.
#SBATCH --job-name=short-pool-full
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-63%16
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_short_pool_full/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_short_pool_full/%A_%a.err
set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_short_pool_full data/processed/pooled_embeddings_v3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

shard=$((SLURM_ARRAY_TASK_ID / 2))
model_id=$((SLURM_ARRAY_TASK_ID % 2))
case "$model_id" in
  0) model=roberta; max_length=512; model_name=roberta ;;
  1) model=bge_m3; max_length=1000; model_name=bge_m3 ;;
  *) echo "invalid model id" >&2; exit 2 ;;
esac
input="data/processed/cleaned/prompt_v3_fixed_parts/shard-${shard}"
base="data/processed/pooled_embeddings_v3/shard-${shard}/${model_name}"
mkdir -p "$base"

for variant in short masked_short; do
  /usr/bin/python3 scripts/run_short_pooled_embeddings.py \
    "$input" \
    --output-dir "$base/$variant" \
    --model "$model" \
    --variant "$variant" \
    --batch-size 4 \
    --max-length "$max_length"
done
