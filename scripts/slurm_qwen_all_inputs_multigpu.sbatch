#!/bin/bash
# One independent Qwen3-Embedding:8b process per physical GPU.
#SBATCH --job-name=cninfo-qwen-mgpu
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --array=0-4%5
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_qwen_mgpu/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_qwen_mgpu/%A_%a.err
set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_qwen_mgpu data/processed/embeddings/all_inputs/qwen3_embedding_8b_multigpu

# c018-10gpu has no Slurm GRES configuration; isolate one physical GPU manually.
GPU_ID=${SLURM_ARRAY_TASK_ID}
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS=/home/gaozh/.ollama/models
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=30m
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
OLLAMA=/home/gaozh/ollama-new/bin/ollama
inputs=(text_plain text_input_2_short text_input_3_long text_input_5_masked_short text_input_6_masked_long)
input=${inputs[$SLURM_ARRAY_TASK_ID]}
out=data/processed/embeddings/all_inputs/qwen3_embedding_8b_multigpu/${input}
mkdir -p "$out"
cleanup(){ [[ -n "${OLLAMA_PID:-}" ]] && kill "$OLLAMA_PID" 2>/dev/null || true; }
trap cleanup EXIT
"$OLLAMA" serve >"logs/slurm_qwen_mgpu/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.ollama.log" 2>&1 &
OLLAMA_PID=$!
for _ in $(seq 1 90); do
  if curl -fsS "$OLLAMA_HOST/api/tags" >/dev/null; then break; fi
  sleep 1
done
curl -fsS "$OLLAMA_HOST/api/tags" >/dev/null
/usr/bin/python3 scripts/run_text_representations.py data/processed/cleaned/prompt_v2_masked_parts \
  --output-dir "$out" --text-column "$input" --ollama \
  --ollama-model qwen3-embedding:8b --ollama-base-url http://127.0.0.1:11435 \
  --max-input-chars 0
