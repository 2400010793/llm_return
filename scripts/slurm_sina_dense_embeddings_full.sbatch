#!/bin/bash
#SBATCH --job-name=sina-embed-full
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/team/llm_return/logs/slurm_sina_embeddings/%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_sina_embeddings/%j.err

set -euo pipefail
cd /home/team/llm_return

input=${INPUT:-/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cleaned/sina_single_stock_clean.parquet}
release=${RELEASE:-/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/embeddings/single_stock}
stage=/tmp/sina-embed-full-${SLURM_JOB_ID}
host=127.0.0.1:11447
ollama=/home/team/ollama-new/bin/ollama

mkdir -p logs/slurm_sina_embeddings "$stage" "$release"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

install_model() {
  local model="$1"
  local destination="$release/$model"
  local incoming="$release/.${model}.${SLURM_JOB_ID}.release"
  if [[ -e "$destination" || -e "$incoming" ]]; then
    echo "refusing to overwrite existing embedding directory: $destination" >&2
    return 1
  fi
  cp -a "$stage/$model" "$incoming"
  mv "$incoming" "$destination"
}

.venv/bin/python scripts/run_text_representations.py "$input" \
  --output-dir "$stage/chinese_roberta" --text-column text --max-input-chars 12000 \
  --roberta --roberta-model /home/team/llm_return/models/chinese-roberta-wwm-ext \
  --transformer-batch-size 32 --max-length 512
.venv/bin/python scripts/audit_embedding_matrix.py \
  --input "$input" --matrix "$stage/chinese_roberta/chinese_roberta.npy" \
  --metadata "$stage/chinese_roberta/metadata.parquet" \
  --output "$stage/chinese_roberta/audit.json" --expected-dimensions 768 \
  --max-input-chars 12000 --allow-duplicate-vectors
install_model chinese_roberta

.venv/bin/python scripts/run_text_representations.py "$input" \
  --output-dir "$stage/bge_m3" --text-column text --max-input-chars 12000 \
  --bge-m3 --bge-m3-model /home/team/llm_return/models/bge-m3 \
  --transformer-batch-size 16 --max-length 512
.venv/bin/python scripts/audit_embedding_matrix.py \
  --input "$input" --matrix "$stage/bge_m3/bge_m3.npy" \
  --metadata "$stage/bge_m3/metadata.parquet" \
  --output "$stage/bge_m3/audit.json" --expected-dimensions 1024 \
  --max-input-chars 12000 --allow-duplicate-vectors
install_model bge_m3

export OLLAMA_HOST="$host"
export OLLAMA_MODELS=/home/team/.ollama/models
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=10m
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
  --output-dir "$stage/qwen3_embedding_8b" --text-column text --max-input-chars 12000 \
  --ollama --ollama-model qwen3-embedding:8b --ollama-base-url "http://${host}"
.venv/bin/python scripts/audit_embedding_matrix.py \
  --input "$input" --matrix "$stage/qwen3_embedding_8b/ollama.npy" \
  --metadata "$stage/qwen3_embedding_8b/metadata.parquet" \
  --output "$stage/qwen3_embedding_8b/audit.json" --expected-dimensions 4096 \
  --max-input-chars 12000 --allow-duplicate-vectors
install_model qwen3_embedding_8b

rm -rf "$stage"
