#!/bin/bash
#SBATCH --job-name=cpu-transformer-probe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_cpu_probe/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_cpu_probe/%A_%a.err
set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_cpu_probe
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-/home/gaozh/.cache/huggingface}
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
case "$SLURM_ARRAY_TASK_ID" in
  0) name=roberta; model=/home/gaozh/llm_return/models/chinese-roberta-wwm-ext ;;
  1) name=bge_m3; model=/home/gaozh/llm_return/models/bge-m3 ;;
esac
/usr/bin/python3 scripts/probe_cpu_transformer_speed.py \
  --name "$name" --model-path "$model" \
  --input data/processed/cleaned/prompt_v2_masked_parts/part-00000.jsonl \
  --rows 256 --batch-size 8 --max-length 256
