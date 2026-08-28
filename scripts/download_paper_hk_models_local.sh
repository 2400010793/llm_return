#!/bin/bash
# Download the paper's China-HK encoders directly, outside Slurm.
set -euo pipefail
cd /home/team/llm_return

MODEL_ROOT=${PAPER_HK_MODEL_ROOT:-/home/team/.cache/huggingface/paper_hk_models}
mkdir -p "${MODEL_ROOT}" logs/direct_downloads
exec 9>"${MODEL_ROOT}/.download.lock"
if ! flock -n 9; then
  echo "Another paper-HK model download already owns ${MODEL_ROOT}/.download.lock"
  exit 0
fi

export HF_HOME=/home/team/.cache/huggingface
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

exec /usr/bin/python3 scripts/prefetch_paper_hk_models.py \
  --model-root "${MODEL_ROOT}" \
  --models xlm_roberta_large \
  --retries 30 \
  --retry-delay 20 \
  --max-workers 1
