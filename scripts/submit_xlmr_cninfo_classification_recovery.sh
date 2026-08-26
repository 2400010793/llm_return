#!/bin/bash
set -euo pipefail
cd /home/gaozh/llm_return

manifest=configs/generated/pooled_xlm_roberta_large_plain_paper_hk.tsv
panel=data/processed/cninfo_full_classification_panel.parquet
output=reports/classification/pooled_embeddings/paper_hk

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name xlmr-cninfo-classification-recovery \
    --purpose "Run the four missing XLM-R full-mean classification cells on the complete CNINFO panel" \
    --input data/processed/pooled_paper_hk_embeddings_v1 \
    --input "${panel}" \
    --output "${manifest}" \
    --output "${output}" \
    --related-file scripts/build_pooled_classification_manifest.py \
    --related-file scripts/slurm_pooled_embedding_classification.sbatch \
    --tag xlmr --tag cninfo --tag classification --tag recovery --tag memory-safe --seed 42 \
    -- bash "$0" "$@"
fi

.venv/bin/python scripts/build_pooled_classification_manifest.py \
  --phase paper_hk \
  --models xlm_roberta_large \
  --variants plain \
  --expected-rows 350577 \
  --output "${manifest}"

tasks=$(($(wc -l < "${manifest}") - 1))
if (( tasks != 4 )); then
  echo "expected 4 pending XLM-R classification cells, found ${tasks}" >&2
  exit 2
fi

job_id=$(.venv/bin/python scripts/task_tracker.py child-submit \
  --task-id "${TASK_RECORD_ID}" -- \
  sbatch --parsable --cpus-per-task=4 --mem=32G --array="0-$((tasks - 1))%2" \
  --export="ALL,MANIFEST=${manifest},PANEL=${panel}" \
  scripts/slurm_pooled_embedding_classification.sbatch)
echo "classification=${job_id} tasks=${tasks} concurrency=2 memory_per_task=32G"
