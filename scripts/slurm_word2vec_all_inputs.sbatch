#!/bin/bash
#SBATCH --job-name=cninfo-w2v
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-4%2
#SBATCH --output=/home/team/llm_return/logs/slurm_word2vec_all/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_word2vec_all/%A_%a.err
set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_word2vec_all data/processed/embeddings/all_inputs/word2vec
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
inputs=(text_plain text_input_2_short text_input_3_long text_input_5_masked_short text_input_6_masked_long)
input=${inputs[$SLURM_ARRAY_TASK_ID]}
/usr/bin/python3 scripts/run_text_representations.py data/processed/cleaned/prompt_v2_masked_parts \
  --output-dir "data/processed/embeddings/all_inputs/word2vec/${input}" --text-column "$input" --word2vec --max-input-chars 0