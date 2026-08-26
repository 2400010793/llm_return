#!/bin/bash
# Audit one embedding model as soon as all of its clean prompt-token tasks finish.
#SBATCH --job-name=prompt-token-model-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_prompt_token_cls_clean/audit_%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_prompt_token_cls_clean/audit_%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${MODEL:?export MODEL=roberta or bge_m3}"
: "${VARIANT:?export VARIANT=short or masked_short}"

ROOT=${ROOT:-data/processed/prompt_token_embeddings_2010_2026_clean_v1}
OUTPUT=${OUTPUT:-reports/cninfo_prompt_token_embeddings_2010_2026_clean_v1_${MODEL}_${VARIANT}_audit.json}
mkdir -p logs/slurm_prompt_token_cls_clean reports
for attempt in $(seq 1 4320); do
  completed=$(find "${ROOT}" -path "*/${MODEL}/${VARIANT}/COMPLETED" -type f | wc -l)
  if [[ "${completed}" -eq 64 ]]; then
    break
  fi
  if (( attempt % 10 == 0 )); then
    echo "waiting for ${MODEL}/${VARIANT}: ${completed}/64"
  fi
  sleep 60
done
[[ "${completed}" -eq 64 ]] || {
  echo "timed out waiting for ${MODEL}/${VARIANT}: ${completed}/64" >&2
  exit 2
}
.venv/bin/python scripts/audit_cninfo_prompt_tokens_2010_2026.py \
  --root "${ROOT}" --rows 903665 --num-shards 64 \
  --models "${MODEL}" --variants "${VARIANT}" --output "${OUTPUT}"
