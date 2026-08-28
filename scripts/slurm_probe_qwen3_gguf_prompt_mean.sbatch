#!/bin/bash
#SBATCH --job-name=qwen3-gguf-readout-probe
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/qwen3_gguf_probe/%j_readout.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/qwen3_gguf_probe/%j_readout.err
set -euo pipefail
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}
MODEL=/home/team/.ollama/models/blobs/sha256-3fcd3febec8b3fd64435204db75bf0dd73b91e8d0661e0331acfe7e7c3120b85
INPUT=/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/inputs/shard-0
OUT=/data/alpha_team2/shares/llm_return/data/interim/qwen3_gguf_prompt_mean_probe
rm -rf "$OUT"
cd /home/team/llm_return
.venv/bin/python scripts/run_qwen3_gguf_prompt_embeddings.py "$INPUT" \
  --model "$MODEL" --prompt '分析股票收益。' --variant short --rows 1 \
  --max-length 500 --output-dir "$OUT"