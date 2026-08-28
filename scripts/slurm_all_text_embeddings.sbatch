#!/bin/bash
#SBATCH --job-name=cninfo-embed
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --array=0-9%3
#SBATCH --output=/home/team/llm_return/logs/slurm_all_embeddings/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_all_embeddings/%A_%a.err
set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_all_embeddings data/processed/embeddings/all_inputs
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME=${HF_HOME:-/home/team/.cache/huggingface}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

inputs=(text_plain text_input_2_short text_input_3_long text_input_5_masked_short text_input_6_masked_long)
input=${inputs[$((SLURM_ARRAY_TASK_ID % 5))]}
model_id=$((SLURM_ARRAY_TASK_ID / 5))
models=(roberta bge_m3)
model=${models[$model_id]}
case "$model" in
  roberta) flag=--roberta; model_arg=--roberta-model; model_path=/home/team/llm_return/models/chinese-roberta-wwm-ext; name=chinese_roberta ;;
  bge_m3) flag=--bge-m3; model_arg=--bge-m3-model; model_path=/home/team/llm_return/models/bge-m3; name=bge_m3 ;;
esac
/usr/bin/python3 scripts/run_text_representations.py data/processed/cleaned/prompt_v2_masked_parts \
  --output-dir "data/processed/embeddings/all_inputs/${name}/${input}" --text-column "$input" \
  "$flag" "$model_arg" "$model_path" --transformer-batch-size 16 --max-length 512 --max-input-chars 0