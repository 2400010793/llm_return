#!/bin/bash
# Download and smoke-test candidate FinBERT checkpoints; this does not generate embeddings.
set -euo pipefail
cd /home/gaozh/llm_return

MODELS=${MODELS:-yiyanghkust/finbert-tone-chinese,hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2,ProsusAI/finbert}
MAX_LENGTH=${MAX_LENGTH:-256}
OUTPUT=${OUTPUT:-reports/finbert/finbert_deployment_probe.json}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name finbert-deployment-probe-v1 \
    --purpose "Download candidate FinBERT checkpoints and run classification-only smoke tests; do not generate embeddings." \
    --output "${OUTPUT}" \
    --related-file scripts/probe_finbert_deployment.py \
    --related-file scripts/slurm_finbert_deployment_probe.sbatch \
    --related-file scripts/submit_finbert_deployment_probe.sh \
    --tag finbert --tag chinese-finance --tag deployment-probe --tag no-embeddings \
    -- bash "$0" "$@"
fi

mkdir -p logs/slurm_finbert_probe reports/finbert
.venv/bin/python -m py_compile scripts/probe_finbert_deployment.py
bash -n scripts/slurm_finbert_deployment_probe.sbatch "$0"

job_id=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable \
      --export="ALL,MODELS=${MODELS},MAX_LENGTH=${MAX_LENGTH},OUTPUT=${OUTPUT}" \
      scripts/slurm_finbert_deployment_probe.sbatch
)
echo "finbert_deployment_probe=${job_id}"
echo "models=${MODELS}"
echo "output=${OUTPUT}"
echo "TASK_RECORD_ID=${TASK_RECORD_ID}"
