#!/bin/bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
cd "${REPO_ROOT}"

START_DATE=${START_DATE:-2026-08-11}
END_DATE=${END_DATE:-2026-08-18}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-${START_DATE//-/}_${END_DATE//-/}}
OUTPUT_DIR=${OUTPUT_DIR:-data/interim/cninfo_increment_${OUTPUT_SUFFIX}}
RAW_DIR=${RAW_DIR:-data/raw/cninfo_increment_${OUTPUT_SUFFIX}}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-reports/cninfo_increment_${OUTPUT_SUFFIX}_audit.json}
STOCKS_FILE=${STOCKS_FILE:-data/stock_universe_cninfo_current_with_returns.csv}
CONCURRENCY=${CONCURRENCY:-20}
SNAPSHOT_FROM_TASK=${SNAPSHOT_FROM_TASK:-20260817T102235Z-simple-states-aligned-pearson-eval-v1-1750506}
EXPECTED_STOCKS=$(awk 'NR > 1 && $0 !~ /,0$/ {count++} END {print count+0}' "${STOCKS_FILE}")

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name "cninfo-increment-${OUTPUT_SUFFIX}" \
    --purpose "Collect and audit ${EXPECTED_STOCKS}-stock CNINFO announcements from ${START_DATE} through ${END_DATE}" \
    --input "${STOCKS_FILE}" \
    --output "${OUTPUT_DIR}" \
    --output "${RAW_DIR}" \
    --output "${AUDIT_OUTPUT}" \
    --related-file scripts/collect_cninfo_announcements.py \
    --related-file scripts/run_cninfo_paper_collection.py \
    --related-file scripts/audit_cninfo_incremental_collection.py \
    --related-file scripts/slurm_cninfo_incremental_collection.sbatch \
    --related-file scripts/slurm_audit_cninfo_incremental_collection.sbatch \
    --related-file scripts/submit_cninfo_incremental_collection.sh \
    --snapshot-from-task "${SNAPSHOT_FROM_TASK}" \
    -- bash "$0" "$@"
fi

case "${CONCURRENCY}" in
  ''|*[!0-9]*) echo "CONCURRENCY must be a positive integer" >&2; exit 2 ;;
esac
(( CONCURRENCY >= 1 && CONCURRENCY <= 32 )) || {
  echo "CONCURRENCY must be between 1 and 32" >&2
  exit 2
}
[[ "${START_DATE}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { echo "invalid START_DATE" >&2; exit 2; }
[[ "${END_DATE}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { echo "invalid END_DATE" >&2; exit 2; }

.venv/bin/python -m py_compile \
  scripts/run_cninfo_paper_collection.py \
  scripts/collect_cninfo_announcements.py \
  scripts/audit_cninfo_incremental_collection.py
bash -n \
  scripts/slurm_cninfo_incremental_collection.sbatch \
  scripts/slurm_audit_cninfo_incremental_collection.sbatch
mkdir -p \
  logs/slurm_cninfo_incremental \
  logs/slurm_cninfo_incremental_audit \
  "${OUTPUT_DIR}" "${RAW_DIR}"

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

exports="ALL,REPO_ROOT=${REPO_ROOT},STOCKS_FILE=${STOCKS_FILE},EXPECTED_STOCKS=${EXPECTED_STOCKS},START_DATE=${START_DATE},END_DATE=${END_DATE},OUTPUT_SUFFIX=${OUTPUT_SUFFIX},OUTPUT_DIR=${OUTPUT_DIR},RAW_DIR=${RAW_DIR},AUDIT_OUTPUT=${AUDIT_OUTPUT}"
chunk_jobs=()
offset=0
previous_job=""
while (( offset < EXPECTED_STOCKS )); do
  remaining=$((EXPECTED_STOCKS - offset))
  (( remaining > 1000 )) && chunk_size=1000 || chunk_size=${remaining}
  submit_args=(
    --parsable
    --array="0-$((chunk_size - 1))%${CONCURRENCY}"
    --export="${exports},STOCK_INDEX_OFFSET=${offset}"
  )
  if [[ -n "${previous_job}" ]]; then
    submit_args+=(--dependency="afterany:${previous_job}")
  fi
  collection_job=$(tracked_sbatch "${submit_args[@]}" scripts/slurm_cninfo_incremental_collection.sbatch)
  chunk_jobs+=("${collection_job}")
  previous_job=${collection_job}
  offset=$((offset + chunk_size))
done
audit_job=$(tracked_sbatch --parsable \
  --dependency="afterany:${previous_job}" \
  --export="${exports}" \
  scripts/slurm_audit_cninfo_incremental_collection.sbatch)

printf 'collection_jobs=%s tasks=%s concurrency=%s range=%s..%s\n' \
  "${chunk_jobs[*]}" "${EXPECTED_STOCKS}" "${CONCURRENCY}" "${START_DATE}" "${END_DATE}"
printf 'audit_job=%s dependency=afterany:%s\n' "${audit_job}" "${previous_job}"
