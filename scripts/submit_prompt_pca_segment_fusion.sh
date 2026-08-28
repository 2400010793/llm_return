#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
MANIFEST=${MANIFEST:-configs/generated/prompt_pca_segment_fusion_roberta_seed42.tsv}
FEATURE_ROOT=${FEATURE_ROOT:-data/processed/prompt_pca_segment_features_2010_2026_clean_v1}
REPORT_ROOT=${REPORT_ROOT:-reports/classification/prompt_pca_segment_fusion}
MAX_PARALLEL=${MAX_PARALLEL:-6}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name prompt-pca-segment-fusion-roberta \
    --purpose "Test training-only Prompt-token PCA128 concatenated with title, body, or both under clean rolling next-day classification" \
    --input "${PANEL}" \
    --input data/processed/pooled_embeddings_2010_2026_roberta_v1 \
    --input data/processed/prompt_token_pca_2010_2026_clean_v1/roberta \
    --output "${FEATURE_ROOT}" \
    --output "${REPORT_ROOT}" \
    --output "${MANIFEST}" \
    --related-file scripts/compose_prompt_pca_segment_features.py \
    --related-file scripts/run_precomputed_embedding_classification.py \
    --related-file scripts/slurm_compose_prompt_pca_segment_features.sbatch \
    --related-file scripts/slurm_prompt_pca_segment_classification.sbatch \
    --related-file scripts/submit_prompt_pca_segment_fusion.sh \
    --related-file configs/generated/prompt_pca_segment_fusion_roberta_seed42.tsv \
    --related-file tests/test_prompt_pca_segment_features.py \
    --tag cninfo --tag roberta --tag prompt-pca --tag segment-fusion \
    --tag classification --tag next-day --tag cpu-only --seed 42 \
    -- bash "$0" "$@"
fi

mkdir -p logs/slurm_prompt_pca_fusion logs/slurm_prompt_pca_fusion_cls "${REPORT_ROOT}"
test -s "${PANEL}"
test -s "${MANIFEST}"
for variant in short masked_short; do
  test -s "data/processed/prompt_token_pca_2010_2026_clean_v1/roberta/${variant}/pca_128.npy"
  test -s "data/processed/prompt_token_pca_2010_2026_clean_v1/roberta/${variant}/summary.json"
done
.venv/bin/python -m py_compile \
  scripts/compose_prompt_pca_segment_features.py \
  scripts/run_precomputed_embedding_classification.py
bash -n \
  scripts/slurm_compose_prompt_pca_segment_features.sbatch \
  scripts/slurm_prompt_pca_segment_classification.sbatch \
  scripts/submit_prompt_pca_segment_fusion.sh

tasks=$(($(wc -l < "${MANIFEST}") - 1))
if [[ "${tasks}" -ne 12 ]]; then
  echo "Expected 12 classification tasks, found ${tasks}" >&2
  exit 2
fi

compose_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --array=0-1%2 \
      --export="ALL,PANEL=${PANEL},OUTPUT_ROOT=${FEATURE_ROOT}" \
      scripts/slurm_compose_prompt_pca_segment_features.sbatch
)
classification_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency="afterok:${compose_job}" \
      --array="0-$((tasks - 1))%${MAX_PARALLEL}" \
      --export="ALL,PANEL=${PANEL},MANIFEST=${MANIFEST}" \
      scripts/slurm_prompt_pca_segment_classification.sbatch
)

echo "prompt_pca_segment_compose=${compose_job} variants=2 memory=96G"
echo "prompt_pca_segment_classification=${classification_job} tasks=${tasks} max_parallel=${MAX_PARALLEL} memory=128G dependency=afterok:${compose_job}"
