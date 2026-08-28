#!/bin/bash
#SBATCH --job-name=qwen-cand-probe
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/team/llm_return/logs/slurm_ollama_candidates_1000_probe/%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_ollama_candidates_1000_probe/%j.err

set -euo pipefail
REPO_ROOT=/home/team/llm_return
cd "${REPO_ROOT}"
mkdir -p logs/slurm_ollama_candidates_1000_probe data/interim/qwen_candidates_1000_probe

export CUDA_VISIBLE_DEVICES=5
export OLLAMA_HOST=127.0.0.1:11440
export OLLAMA_MODELS=/home/team/.ollama/models
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=30m
OLLAMA=/home/team/ollama-new/bin/ollama
cleanup(){ [[ -n "${OLLAMA_PID:-}" ]] && kill "${OLLAMA_PID}" 2>/dev/null || true; }
trap cleanup EXIT
"${OLLAMA}" serve >"logs/slurm_ollama_candidates_1000_probe/${SLURM_JOB_ID}.ollama.log" 2>&1 &
OLLAMA_PID=$!
for _ in $(seq 1 120); do
  curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null && break
  sleep 1
done
curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null

"${REPO_ROOT}/.venv/bin/python" scripts/run_ollama_candidate_shard.py \
  data/processed/cleaned/prompt_v2_masked_parts/part-00000.jsonl \
  --output-dir data/interim/qwen_candidates_1000_probe \
  --candidates short,masked_short,plain \
  --model qwen3-embedding:8b \
  --base-url "http://${OLLAMA_HOST}" \
  --max-tokens 1000 \
  --batch-size 8 \
  --timeout 600 \
  --max-rows 256
