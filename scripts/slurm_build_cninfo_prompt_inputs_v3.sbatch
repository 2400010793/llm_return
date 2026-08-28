#!/bin/bash
# Rebuild fixed natural-language short prompts and natural-language masks.
# Each array task scans a disjoint modulo shard; the 32 tasks write separate
# output files and therefore do not contend on one JSONL file.
#SBATCH --job-name=cninfo-prompt-v3
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-31%32
#SBATCH --output=/home/team/llm_return/logs/slurm_prompt_v3/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_prompt_v3/%A_%a.err
set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_prompt_v3 data/processed/cleaned/prompt_v3_fixed_parts
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

INPUT=data/processed/cleaned/cninfo_announcements_final.jsonl
OUT=data/processed/cleaned/prompt_v3_fixed_parts
/usr/bin/python3 scripts/build_cninfo_masked_parts.py \
  --input "$INPUT" \
  --output-dir "$OUT/shard-${SLURM_ARRAY_TASK_ID}" \
  --manifest "$OUT/shard-${SLURM_ARRAY_TASK_ID}.manifest.json" \
  --part-size 5000 \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards 32