#!/bin/bash
#SBATCH --job-name=cninfo-qwen
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --array=0-4%1
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_qwen_all/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_qwen_all/%A_%a.err
set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_qwen_all data/processed/embeddings/all_inputs/qwen3_embedding_8b
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=/home/gaozh/.ollama/models
export OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=30m
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
OLLAMA=/home/gaozh/ollama-new/bin/ollama
inputs=(text_plain text_input_2_short text_input_3_long text_input_5_masked_short text_input_6_masked_long)
input=${inputs[$SLURM_ARRAY_TASK_ID]}
mkdir -p "data/processed/embeddings/all_inputs/qwen3_embedding_8b/${input}"
cleanup(){ [[ -n "${OLLAMA_PID:-}" ]] && kill "$OLLAMA_PID" 2>/dev/null || true; }
trap cleanup EXIT
"$OLLAMA" serve >"logs/slurm_qwen_all/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.ollama.log" 2>&1 &
OLLAMA_PID=$!
for _ in $(seq 1 90); do curl -fsS "$OLLAMA_HOST/api/tags" >/dev/null && break; sleep 1; done
/usr/bin/python3 scripts/run_text_representations.py data/processed/cleaned/prompt_v2_masked_parts \
  --output-dir "data/processed/embeddings/all_inputs/qwen3_embedding_8b/${input}" \
  --text-column "$input" --ollama --ollama-model qwen3-embedding:8b --ollama-base-url http://127.0.0.1:11435 --max-input-chars 0