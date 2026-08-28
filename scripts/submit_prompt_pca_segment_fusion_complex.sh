#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
MANIFEST=${MANIFEST:-configs/generated/prompt_pca_segment_fusion_complex_roberta_seed42.tsv}
FEATURE_ROOT=${FEATURE_ROOT:-data/processed/prompt_pca_segment_features_2010_2026_clean_v1/roberta}
REPORT_ROOT=${REPORT_ROOT:-reports/classification/prompt_pca_segment_fusion_complex}
MAX_PARALLEL=${MAX_PARALLEL:-6}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name prompt-pca-segment-fusion-complex-roberta \
    --purpose "Test MLP, XGBoost, and histogram gradient boosting on the strongest training-only Prompt-PCA plus pooled-segment representations" \
    --input "${PANEL}" \
    --input "${FEATURE_ROOT}" \
    --output "${REPORT_ROOT}" \
    --output "${MANIFEST}" \
    --related-file scripts/run_precomputed_embedding_classification.py \
    --related-file scripts/run_pooled_embedding_classification.py \
    --related-file scripts/slurm_prompt_pca_segment_classification.sbatch \
    --related-file scripts/submit_prompt_pca_segment_fusion_complex.sh \
    --related-file configs/generated/prompt_pca_segment_fusion_complex_roberta_seed42.tsv \
    --tag cninfo --tag roberta --tag prompt-pca --tag segment-fusion \
    --tag complex-classifier --tag next-day --tag cpu-only --seed 42 \
    -- bash "$0" "$@"
fi

mkdir -p logs/slurm_prompt_pca_fusion_cls "${REPORT_ROOT}"
test -s "${PANEL}"
test -s "${MANIFEST}"
for variant in short masked_short; do
  test -s "${FEATURE_ROOT}/${variant}/COMPLETED"
  test -s "${FEATURE_ROOT}/${variant}/metadata.parquet"
done
bash -n \
  scripts/slurm_prompt_pca_segment_classification.sbatch \
  scripts/submit_prompt_pca_segment_fusion_complex.sh

tasks=$(($(wc -l < "${MANIFEST}") - 1))
if [[ "${tasks}" -ne 12 ]]; then
  echo "Expected 12 complex classification tasks, found ${tasks}" >&2
  exit 2
fi

job_id=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --array="0-$((tasks - 1))%${MAX_PARALLEL}" \
      --export="ALL,PANEL=${PANEL},MANIFEST=${MANIFEST}" \
      scripts/slurm_prompt_pca_segment_classification.sbatch
)

echo "prompt_pca_segment_complex=${job_id} tasks=${tasks} max_parallel=${MAX_PARALLEL} memory=128G cpu_only=true"
