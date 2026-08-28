#!/bin/bash
#SBATCH --job-name=qwen-probe
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/team/llm_return/logs/slurm_qwen_probe/%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_qwen_probe/%j.err

set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_qwen_probe
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS=/home/team/.ollama/models
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=30m
OLLAMA=/home/team/ollama-new/bin/ollama

cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]]; then kill "${OLLAMA_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT

printf '%s\n' "host=$(hostname)" "cuda_visible=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$OLLAMA" serve >"logs/slurm_qwen_probe/${SLURM_JOB_ID}.ollama.log" 2>&1 &
OLLAMA_PID=$!

for _ in $(seq 1 60); do
  if curl -fsS "$OLLAMA_HOST/api/tags" >/tmp/qwen_tags.json; then break; fi
  sleep 1
done
cat /tmp/qwen_tags.json

/usr/bin/python3 - <<'PY'
import json, urllib.request
base = "http://127.0.0.1:11435"
model = "qwen3-embedding:8b"
payload = json.dumps({"model": model, "input": ["这是一个Qwen embedding测试。"]}).encode()
req = urllib.request.Request(base + "/api/embed", data=payload, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=300) as response:
    result = json.loads(response.read().decode())
embeddings = result.get("embeddings")
if not embeddings or not embeddings[0]:
    raise RuntimeError(f"empty embedding response: {result}")
print(json.dumps({"model": model, "rows": len(embeddings), "dimensions": len(embeddings[0])}, ensure_ascii=False))
PY
