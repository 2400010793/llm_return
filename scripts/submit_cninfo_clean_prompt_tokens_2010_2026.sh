#!/bin/bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
cd "${REPO_ROOT}"

OUTPUT_ROOT=${OUTPUT_ROOT:-data/processed/prompt_token_embeddings_2010_2026_clean_v1}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-reports/cninfo_prompt_token_embeddings_2010_2026_clean_v1_audit.json}
CONCURRENCY=${CONCURRENCY:-16}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name cninfo-clean-prompt-tokens-2010-2026 \
    --purpose "Retain every ordered Prompt-token embedding from audited clean_v1 CNINFO inputs" \
    --input data/processed/cleaned/cninfo_prompt_bundle_existing_2018_2026_clean_v1/prompt_v3_fixed_parts \
    --input data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017_clean_v1/prompt_v3_fixed_parts \
    --output "${OUTPUT_ROOT}" \
    --output "${AUDIT_OUTPUT}" \
    --related-file scripts/run_prompt_token_embeddings_v5.py \
    --related-file scripts/audit_cninfo_prompt_tokens_2010_2026.py \
    --related-file scripts/slurm_cninfo_clean_prompt_tokens_2010_2026.sbatch \
    --related-file scripts/slurm_audit_cninfo_prompt_tokens_2010_2026.sbatch \
    -- bash "$0" "$@"
fi

case "${CONCURRENCY}" in
  ''|*[!0-9]*) echo "CONCURRENCY must be a positive integer" >&2; exit 2 ;;
esac
(( CONCURRENCY >= 1 && CONCURRENCY <= 32 )) || {
  echo "CONCURRENCY must be between 1 and 32" >&2
  exit 2
}

.venv/bin/python - <<'PY'
import json
from pathlib import Path
checks = (
    (Path("reports/cninfo_prompt_bundle_existing_2018_2026_clean_v1_audit.json"), 350577),
    (Path("reports/cninfo_prompt_bundle_increment_2010_2017_clean_v1_audit.json"), 553088),
)
for path, rows in checks:
    report = json.loads(path.read_text())
    if report.get("status") != "passed" or report.get("checked_rows") != rows:
        raise SystemExit(f"clean Prompt audit is not valid: {path}")
print("clean Prompt source audits: passed")
PY

.venv/bin/python -m py_compile \
  scripts/run_prompt_token_embeddings_v5.py \
  scripts/audit_cninfo_prompt_tokens_2010_2026.py
bash -n \
  scripts/slurm_cninfo_clean_prompt_tokens_2010_2026.sbatch \
  scripts/slurm_audit_cninfo_prompt_tokens_2010_2026.sbatch
mkdir -p \
  logs/slurm_cninfo_clean_prompt_tokens \
  logs/slurm_cninfo_prompt_token_audit \
  "${OUTPUT_ROOT}"

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

embedding_job=$(tracked_sbatch --parsable \
  --array="0-127%${CONCURRENCY}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}" \
  scripts/slurm_cninfo_clean_prompt_tokens_2010_2026.sbatch)
audit_job=$(tracked_sbatch --parsable \
  --dependency="afterok:${embedding_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},ROOT=${OUTPUT_ROOT},OUTPUT=${AUDIT_OUTPUT}" \
  scripts/slurm_audit_cninfo_prompt_tokens_2010_2026.sbatch)

printf 'embedding_job=%s tasks=128 concurrency=%s\n' "${embedding_job}" "${CONCURRENCY}"
printf 'audit_job=%s dependency=afterok:%s\n' "${audit_job}" "${embedding_job}"
