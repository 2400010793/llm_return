#!/bin/bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
cd "${REPO_ROOT}"

PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
INPUT_ROOT=${INPUT_ROOT:-data/processed/prompt_token_embeddings_2010_2026_clean_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/dynamic_prompt/rolling_cpu_clean_2010_2026}
ROBERTA_AUDIT_JOB=${ROBERTA_AUDIT_JOB:-4335131}
BGE_M3_AUDIT_JOB=${BGE_M3_AUDIT_JOB:-4335138}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name dynamic-prompt-rolling-cpu-clean-2010-2026 \
    --purpose "Compare CPU-only uniform, static, and dynamic prompt-token gates in nine rolling classification folds" \
    --input "${INPUT_ROOT}" --input "${PANEL}" --output "${OUTPUT_ROOT}" \
    --related-file scripts/run_dynamic_prompt_token_gating.py \
    --related-file src/data/prompt_token_embeddings.py \
    --related-file src/models/dynamic_token_gating.py \
    --related-file scripts/slurm_dynamic_prompt_rolling_cpu_clean_2010_2026.sbatch \
    --related-file scripts/summarize_dynamic_prompt_rolling_cpu_clean_2010_2026.py \
    --tag classification --tag prompt-token-gating --tag cpu-only --seed 42 \
    -- bash "$0" "$@"
fi

.venv/bin/python -m py_compile \
  scripts/run_dynamic_prompt_token_gating.py \
  scripts/summarize_dynamic_prompt_rolling_cpu_clean_2010_2026.py
bash -n \
  scripts/slurm_dynamic_prompt_rolling_cpu_clean_2010_2026.sbatch \
  scripts/slurm_summarize_dynamic_prompt_rolling_cpu_clean_2010_2026.sbatch
mkdir -p logs/slurm_dynamic_prompt_cpu_clean "${OUTPUT_ROOT}"

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

for model in roberta bge_m3; do
  if [[ "${model}" == roberta ]]; then
    dependency=${ROBERTA_AUDIT_JOB}
  else
    dependency=${BGE_M3_AUDIT_JOB}
  fi
  gate_job=$(tracked_sbatch --parsable --dependency="afterok:${dependency}" \
    --array="0-26%6" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},MODEL=${model},PANEL=${PANEL},INPUT_ROOT=${INPUT_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}" \
    scripts/slurm_dynamic_prompt_rolling_cpu_clean_2010_2026.sbatch)
  summary_job=$(tracked_sbatch --parsable --dependency="afterok:${gate_job}" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},MODEL=${model},ROOT=${OUTPUT_ROOT},OUTPUT=${OUTPUT_ROOT}/${model}_masked_short_summary.json" \
    scripts/slurm_summarize_dynamic_prompt_rolling_cpu_clean_2010_2026.sbatch)
  printf '%s dependency=%s gate=%s summary=%s\n' \
    "${model}" "${dependency}" "${gate_job}" "${summary_job}"
done
