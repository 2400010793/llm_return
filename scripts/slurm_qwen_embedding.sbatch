#!/bin/bash
#SBATCH --job-name=qwen-embed
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_qwen_embedding/%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_qwen_embedding/%j.err

set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_qwen_embedding data/processed/embeddings/qwen3_embedding_8b
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS=/home/gaozh/.ollama/models
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=30m
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
OLLAMA=/home/gaozh/ollama-new/bin/ollama

cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]]; then kill "${OLLAMA_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT

"$OLLAMA" serve >"logs/slurm_qwen_embedding/${SLURM_JOB_ID}.ollama.log" 2>&1 &
OLLAMA_PID=$!
for _ in $(seq 1 60); do
  if curl -fsS "$OLLAMA_HOST/api/tags" >/tmp/qwen_tags.json; then break; fi
  sleep 1
done

/usr/bin/python3 - <<'PY'
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("data/processed/cninfo_100_classification_panel.parquet")
OUT = Path("data/processed/embeddings/qwen3_embedding_8b")
MODEL = "qwen3-embedding:8b"
BASE = "http://127.0.0.1:11435"
MAX_INPUT_CHARS = 12000
BATCH_SIZE = 8

frame = pd.read_parquet(PANEL).copy()
if "text" not in frame or "document_id" not in frame:
    raise ValueError("panel must contain text and document_id")
texts = [str(value)[:MAX_INPUT_CHARS] if pd.notna(value) else "" for value in frame["text"]]
all_vectors = []
for start in range(0, len(texts), BATCH_SIZE):
    batch = texts[start:start + BATCH_SIZE]
    payload = json.dumps({"model": MODEL, "input": batch}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        BASE + "/api/embed", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.loads(response.read().decode("utf-8"))
    vectors = result.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(batch):
        raise RuntimeError(f"invalid embedding response at rows {start}:{start + len(batch)}")
    all_vectors.extend(vectors)
    if (start // BATCH_SIZE + 1) % 100 == 0:
        print(json.dumps({"processed": start + len(batch), "total": len(texts)}), flush=True)

matrix = np.asarray(all_vectors, dtype=np.float32)
if matrix.shape != (len(frame), 4096):
    raise RuntimeError(f"unexpected embedding shape: {matrix.shape}")
OUT.mkdir(parents=True, exist_ok=True)
np.save(OUT / "embeddings.npy", matrix)
metadata = frame[["document_id", "stock_id", "entry_date"]].copy()
metadata["row_index"] = np.arange(len(metadata), dtype=np.int64)
metadata["text_sha256"] = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
metadata.to_parquet(OUT / "metadata.parquet", index=False)
summary = {
    "model": MODEL,
    "panel": str(PANEL),
    "rows": int(matrix.shape[0]),
    "dimensions": int(matrix.shape[1]),
    "max_input_chars": MAX_INPUT_CHARS,
    "batch_size": BATCH_SIZE,
    "embedding_file": str(OUT / "embeddings.npy"),
    "metadata_file": str(OUT / "metadata.parquet"),
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
