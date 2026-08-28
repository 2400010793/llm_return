#!/bin/bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
cd "${REPO_ROOT}"

TOKEN_ROOT=${TOKEN_ROOT:-data/processed/prompt_token_embeddings_2010_2026_clean_v1}
PCA_ROOT=${PCA_ROOT:-data/processed/prompt_token_pca_2010_2026_clean_v1}
REPORT_ROOT=${REPORT_ROOT:-reports/classification/prompt_tokens_2010_2026_clean_v1}
PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name clean-prompt-token-rolling-classification-2010-2026 \
    --purpose "Run leakage-safe rolling classifiers on ordered clean prompt-token embeddings" \
    --input "${TOKEN_ROOT}" --input "${PANEL}" \
    --output "${PCA_ROOT}" --output "${REPORT_ROOT}" \
    --related-file scripts/run_prompt_token_pca_v5.py \
    --related-file scripts/run_reduced_prompt_classification.py \
    --related-file scripts/audit_cninfo_prompt_tokens_2010_2026.py \
    --related-file scripts/summarize_prompt_token_classification_clean_2010_2026.py \
    --related-file scripts/slurm_audit_prompt_token_model_2010_2026.sbatch \
    --related-file scripts/slurm_prompt_token_pca_clean_2010_2026.sbatch \
    --related-file scripts/slurm_prompt_token_classification_clean_2010_2026.sbatch \
    --tag classification --tag prompt-token --tag rolling --seed 42 \
    -- bash "$0" "$@"
fi

.venv/bin/python -m py_compile \
  scripts/run_prompt_token_pca_v5.py \
  scripts/run_reduced_prompt_classification.py \
  scripts/audit_cninfo_prompt_tokens_2010_2026.py \
  scripts/summarize_prompt_token_classification_clean_2010_2026.py
bash -n \
  scripts/slurm_audit_prompt_token_model_2010_2026.sbatch \
  scripts/slurm_prompt_token_pca_clean_2010_2026.sbatch \
  scripts/slurm_prompt_token_classification_clean_2010_2026.sbatch \
  scripts/slurm_summarize_prompt_token_classification_clean_2010_2026.sbatch

mkdir -p \
  logs/slurm_prompt_token_cls_clean "${PCA_ROOT}" "${REPORT_ROOT}"

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

classification_jobs=()
for model in roberta bge_m3; do
  model_classification_jobs=()
  for variant in short masked_short; do
    audit_output=reports/cninfo_prompt_token_embeddings_2010_2026_clean_v1_${model}_${variant}_audit.json
    audit_job=$(tracked_sbatch --parsable \
      --export="ALL,REPO_ROOT=${REPO_ROOT},MODEL=${model},VARIANT=${variant},ROOT=${TOKEN_ROOT},OUTPUT=${audit_output}" \
      scripts/slurm_audit_prompt_token_model_2010_2026.sbatch)
    pca_job=$(tracked_sbatch --parsable --dependency="afterok:${audit_job}" \
      --array="0-8%4" \
      --export="ALL,REPO_ROOT=${REPO_ROOT},MODEL=${model},VARIANT=${variant},PANEL=${PANEL},INPUT_ROOT=${TOKEN_ROOT},OUTPUT_ROOT=${PCA_ROOT}" \
      scripts/slurm_prompt_token_pca_clean_2010_2026.sbatch)
    classification_job=$(tracked_sbatch --parsable --dependency="afterok:${pca_job}" \
      --array="0-11%8" \
      --export="ALL,REPO_ROOT=${REPO_ROOT},MODEL=${model},VARIANT=${variant},PANEL=${PANEL},PCA_ROOT=${PCA_ROOT},OUTPUT_ROOT=${REPORT_ROOT}" \
      scripts/slurm_prompt_token_classification_clean_2010_2026.sbatch)
    model_classification_jobs+=("${classification_job}")
    classification_jobs+=("${classification_job}")
    printf '%s/%s audit=%s pca=%s classification=%s\n' \
      "${model}" "${variant}" "${audit_job}" "${pca_job}" "${classification_job}"
  done
  summary_job=$(tracked_sbatch --parsable \
    --dependency="afterok:${model_classification_jobs[0]}:${model_classification_jobs[1]}" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},MODEL=${model},ROOT=${REPORT_ROOT},OUTPUT=${REPORT_ROOT}/${model}_summary.json" \
    scripts/slurm_summarize_prompt_token_classification_clean_2010_2026.sbatch)
  printf '%s summary=%s\n' "${model}" "${summary_job}"
done
