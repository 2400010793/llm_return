#!/bin/bash
#SBATCH --job-name=qwen-cand-1k
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-70%5
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_ollama_candidates_1000/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_ollama_candidates_1000/%A_%a.err

set -euo pipefail
REPO_ROOT=/home/gaozh/llm_return
cd "${REPO_ROOT}"
mkdir -p logs/slurm_ollama_candidates_1000 data/processed/embeddings/qwen3_embedding_8b_candidates_1000

# Slurm on this node does not expose GPU GRES. Claim one of the GPUs observed
# free at submission time. A file lock prevents a replacement array task from
# colliding with a slower task that still owns the same physical GPU.
GPU_ID=
GPU_LOCK_FD=
for candidate_gpu in 1 2 4 5 6; do
  exec {lock_fd}>"/tmp/gaozh_qwen_candidate_gpu_${candidate_gpu}.lock"
  if flock -n "${lock_fd}"; then
    GPU_ID=${candidate_gpu}
    GPU_LOCK_FD=${lock_fd}
    break
  fi
  eval "exec ${lock_fd}>&-"
done
if [[ -z "${GPU_ID}" ]]; then
  echo "No configured Qwen GPU lock is available" >&2
  exit 1
fi
PORT=$((11435 + GPU_ID))
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export OLLAMA_HOST=127.0.0.1:${PORT}
export OLLAMA_MODELS=/home/gaozh/.ollama/models
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=30m
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

OLLAMA=/home/gaozh/ollama-new/bin/ollama
cleanup(){ [[ -n "${OLLAMA_PID:-}" ]] && kill "${OLLAMA_PID}" 2>/dev/null || true; }
trap cleanup EXIT
"${OLLAMA}" serve >"logs/slurm_ollama_candidates_1000/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.ollama.log" 2>&1 &
OLLAMA_PID=$!
for _ in $(seq 1 120); do
  curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null && break
  sleep 1
done
curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null

printf -v PART '%05d' "${SLURM_ARRAY_TASK_ID}"
INPUT="data/processed/cleaned/prompt_v2_masked_parts/part-${PART}.jsonl"
OUTPUT="data/processed/embeddings/qwen3_embedding_8b_candidates_1000/part-${PART}"
"${REPO_ROOT}/.venv/bin/python" scripts/run_ollama_candidate_shard.py "${INPUT}" \
  --output-dir "${OUTPUT}" \
  --candidates short,masked_short,plain \
  --model qwen3-embedding:8b \
  --base-url "http://${OLLAMA_HOST}" \
  --max-tokens 1000 \
  --batch-size 8 \
  --timeout 600
