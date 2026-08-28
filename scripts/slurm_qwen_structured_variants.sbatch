#!/bin/bash
#SBATCH --job-name=cninfo-qwen-struct
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
# One Qwen process encodes all variants in one pass over the model.
#SBATCH --array=0-0
#SBATCH --output=/home/team/llm_return/logs/slurm_qwen_struct/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_qwen_struct/%A_%a.err
set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_qwen_struct data/processed/embeddings/qwen3_embedding_8b_structured

# The existing Qwen array uses physical GPUs 0-4. This follow-up uses GPU 5.
GPU_ID=5
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export OLLAMA_HOST=127.0.0.1:11436
export OLLAMA_MODELS=/home/team/.ollama/models
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=30m
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
OLLAMA=/home/team/ollama-new/bin/ollama
out=data/processed/embeddings/qwen3_embedding_8b_structured
mkdir -p "$out"
cleanup(){ [[ -n "${OLLAMA_PID:-}" ]] && kill "$OLLAMA_PID" 2>/dev/null || true; }
trap cleanup EXIT
"$OLLAMA" serve >"logs/slurm_qwen_struct/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.ollama.log" 2>&1 &
OLLAMA_PID=$!
for _ in $(seq 1 90); do
  if curl -fsS "$OLLAMA_HOST/api/tags" >/dev/null; then break; fi
  sleep 1
done
curl -fsS "$OLLAMA_HOST/api/tags" >/dev/null
/usr/bin/python3 scripts/run_multi_variant_embeddings.py \
  data/processed/cleaned/cninfo_announcements_final.jsonl \
  --output-dir "$out" --backend ollama --model qwen3-embedding:8b \
  --base-url http://127.0.0.1:11436 --batch-size 8
