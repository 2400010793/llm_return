#!/bin/bash
#SBATCH --job-name=cninfo-long-v4
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-31%16
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_long_v4/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_long_v4/%A_%a.err
set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_long_v4 data/processed/cleaned/prompt_v4_fixed_long_parts
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
/usr/bin/python3 scripts/build_cninfo_masked_parts.py \
  --input data/processed/cleaned/cninfo_announcements_final.jsonl \
  --output-dir data/processed/cleaned/prompt_v4_fixed_long_parts/shard-${SLURM_ARRAY_TASK_ID} \
  --manifest data/processed/cleaned/prompt_v4_fixed_long_parts/shard-${SLURM_ARRAY_TASK_ID}.manifest.json \
  --part-size 5000 --shard-id "$SLURM_ARRAY_TASK_ID" --num-shards 32
