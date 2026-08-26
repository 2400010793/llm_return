#!/bin/bash
#SBATCH --job-name=qwen3-gguf-probe
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/qwen3_gguf_probe/%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/qwen3_gguf_probe/%j.err
set -euo pipefail
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
MODEL=${MODEL:-/home/gaozh/.ollama/models/blobs/sha256-3fcd3febec8b3fd64435204db75bf0dd73b91e8d0661e0331acfe7e7c3120b85}
GPU_ID=${GPU_ID:--1}
mkdir -p /mnt/lustre3/home/gaozh/llm_return/logs/qwen3_gguf_probe
if (( GPU_ID < 0 )); then
  for candidate in $(seq 0 9); do
    free_candidate=$(nvidia-smi --id=${candidate} --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)
    if (( free_candidate > 32000 )); then GPU_ID=${candidate}; break; fi
  done
fi
(( GPU_ID >= 0 ))
export CUDA_VISIBLE_DEVICES=${GPU_ID}
free=$(nvidia-smi --id=${GPU_ID} --query-gpu=memory.free --format=csv,noheader,nounits)
echo "host=$(hostname) gpu=${GPU_ID} free_mib=${free}"
(( free > 32000 ))
cd /home/gaozh/llm_return
.venv/bin/python scripts/probe_qwen3_gguf_token_embeddings.py --model "${MODEL}" --gpu-id "${GPU_ID}" --max-length 500 --output \
  /mnt/lustre3/home/gaozh/llm_return/data/interim/qwen3_gguf_probe/result.json