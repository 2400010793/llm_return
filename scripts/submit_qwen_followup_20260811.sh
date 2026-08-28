#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name qwen-followup-20260811 \
    --purpose "Extend Qwen screens with available strong classifiers, robust O2O regressors, PCA dimensions, and independent hard-top4 regression controls" \
    --input data/processed/pooled_embeddings_qwen_candidates_1000 \
    --input data/processed/cninfo_full_classification_panel.parquet \
    --input data/processed/cninfo_full_o2o_residual_panel.parquet \
    --input data/processed/prompt_token_embeddings_v5 \
    --output reports/classification/qwen_followup_screen_20260811 \
    --output reports/regression/qwen_followup_screen_20260811 \
    --output reports/dynamic_prompt/o2o_top4_controls_20260811 \
    --related-file configs/generated/qwen_followup_classification_20260811.tsv \
    --related-file configs/generated/qwen_followup_regression_20260811.tsv \
    --related-file configs/generated/prompt_top4_regression_controls_20260811.tsv \
    --related-file scripts/slurm_qwen_classification_screen_20260811.sbatch \
    --related-file scripts/slurm_qwen_regression_screen_20260811.sbatch \
    --related-file scripts/slurm_prompt_topk_overnight_20260811.sbatch \
    --tag qwen --tag next1 --tag o2o --tag strong-classifier --tag robust-regression --tag hard-top4 --seed 42 \
    -- bash "$0" "$@"
fi

QWEN_JOB=3995435
PREPARE_JOB=4015405
CLASSIFICATION_MANIFEST=configs/generated/qwen_followup_classification_20260811.tsv
REGRESSION_MANIFEST=configs/generated/qwen_followup_regression_20260811.tsv
PROMPT_MANIFEST=configs/generated/prompt_top4_regression_controls_20260811.tsv

mkdir -p \
  reports/classification/qwen_followup_screen_20260811 \
  reports/regression/qwen_followup_screen_20260811 \
  reports/dynamic_prompt/o2o_top4_controls_20260811

for manifest in "${CLASSIFICATION_MANIFEST}" "${REGRESSION_MANIFEST}" "${PROMPT_MANIFEST}"; do
  test -s "${manifest}"
done
for script in \
  scripts/slurm_qwen_classification_screen_20260811.sbatch \
  scripts/slurm_qwen_regression_screen_20260811.sbatch \
  scripts/slurm_prompt_topk_overnight_20260811.sbatch; do
  bash -n "${script}"
done

classification_tasks=$(( $(wc -l < "${CLASSIFICATION_MANIFEST}") - 1 ))
regression_tasks=$(( $(wc -l < "${REGRESSION_MANIFEST}") - 1 ))
prompt_tasks=$(( $(wc -l < "${PROMPT_MANIFEST}") - 1 ))

# Qwen consumers require a successfully prepared pooled store.  Each array
# task remains independent, so one classifier/regressor failure cannot cancel
# its siblings.  Prompt-token controls need no Qwen output and therefore use
# afterany only to avoid competing with the current Ollama embedding run.
classification_id=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency=afterok:${PREPARE_JOB} \
      --array="0-$((classification_tasks - 1))%2" \
      --export=ALL,MANIFEST="${CLASSIFICATION_MANIFEST}" \
      scripts/slurm_qwen_classification_screen_20260811.sbatch
)
regression_id=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency=afterok:${PREPARE_JOB} \
      --array="0-$((regression_tasks - 1))%2" \
      --export=ALL,MANIFEST="${REGRESSION_MANIFEST}" \
      scripts/slurm_qwen_regression_screen_20260811.sbatch
)
prompt_id=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency=afterany:${QWEN_JOB} \
      --array="0-$((prompt_tasks - 1))%2" \
      --export=ALL,MANIFEST="${PROMPT_MANIFEST}" \
      scripts/slurm_prompt_topk_overnight_20260811.sbatch
)

echo "submitted qwen_followup_classification=${classification_id} tasks=${classification_tasks} throttle=2 dependency=afterok:${PREPARE_JOB}"
echo "submitted qwen_followup_regression=${regression_id} tasks=${regression_tasks} throttle=2 dependency=afterok:${PREPARE_JOB}"
echo "submitted prompt_top4_controls=${prompt_id} tasks=${prompt_tasks} throttle=2 dependency=afterany:${QWEN_JOB}"
