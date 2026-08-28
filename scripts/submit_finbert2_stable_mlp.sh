#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

MANIFEST=${MANIFEST:-configs/generated/finbert2_stable_mlp.tsv}
PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-data/processed/pooled_finbert2_embeddings_2010_2026_v1}
MAX_PARALLEL=${MAX_PARALLEL:-8}

.venv/bin/python scripts/build_finbert2_stable_mlp_manifest.py "${MANIFEST}"

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name finbert2-stable-mlp-classification \
    --purpose "Focused FinBERT2 classification with five seeds, chronological Accuracy early stopping, burn-in, and constant/cosine schedules" \
    --input "${PANEL}" \
    --input "${EMBEDDING_ROOT}" \
    --input "${MANIFEST}" \
    --output reports/classification/pooled_embeddings/finbert2_stable_mlp \
    --related-file scripts/build_finbert2_stable_mlp_manifest.py \
    --related-file scripts/slurm_finbert2_stable_mlp.sbatch \
    --related-file scripts/run_pooled_embedding_classification.py \
    --tag finbert2 --tag classification --tag seed-stability --tag annealing \
    -- bash "$0" "$@"
fi

tasks=$(( $(wc -l < "${MANIFEST}") - 1 ))
[[ "${tasks}" -eq 30 ]] || { echo "expected 30 tasks, found ${tasks}" >&2; exit 2; }
mkdir -p logs/slurm_finbert2_stable_mlp reports/classification/pooled_embeddings/finbert2_stable_mlp

job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --array="0-$((tasks - 1))%${MAX_PARALLEL}" \
      --export="ALL,MANIFEST=${MANIFEST},PANEL=${PANEL},EMBEDDING_ROOT=${EMBEDDING_ROOT},EXPECTED_ROWS=903665" \
      scripts/slurm_finbert2_stable_mlp.sbatch
)
echo "submitted FinBERT2 stable MLP array=${job} tasks=${tasks} max_parallel=${MAX_PARALLEL}"
