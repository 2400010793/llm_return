#!/bin/bash
#SBATCH --job-name=sina-qwen100
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_sina_embeddings/%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_sina_embeddings/%j.err

set -euo pipefail
cd /home/gaozh/llm_return

input=/home/gaozh/news_content_quality_20260812/validation_100/random_100.parquet
output=/home/gaozh/news_content_quality_20260812/validation_100/embeddings/qwen3_embedding_8b
host=127.0.0.1:11447

mkdir -p logs/slurm_sina_embeddings "$output"
export CUDA_VISIBLE_DEVICES=1
export OLLAMA_HOST="$host"
export OLLAMA_MODELS=/home/gaozh/.ollama/models
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=10m

ollama=/home/gaozh/ollama-new/bin/ollama
cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]]; then
    kill "$OLLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"$ollama" serve >"logs/slurm_sina_embeddings/${SLURM_JOB_ID}.ollama.log" 2>&1 &
OLLAMA_PID=$!
for _ in $(seq 1 90); do
  if curl -fsS "http://${host}/api/tags" >/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS "http://${host}/api/tags" >/dev/null

.venv/bin/python scripts/run_text_representations.py "$input" \
  --output-dir "$output" \
  --text-column text \
  --max-input-chars 12000 \
  --ollama \
  --ollama-model qwen3-embedding:8b \
  --ollama-base-url "http://${host}"

.venv/bin/python scripts/audit_embedding_matrix.py \
  --input "$input" \
  --matrix "$output/ollama.npy" \
  --metadata "$output/metadata.parquet" \
  --output "$output/audit.json" \
  --expected-dimensions 4096 \
  --max-input-chars 12000
