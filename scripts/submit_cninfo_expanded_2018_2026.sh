#!/bin/bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
cd "${REPO_ROOT}"

STOCKS_FILE=${STOCKS_FILE:-data/stock_universe_cninfo_current_with_returns.csv}
OUTPUT_DIR=${OUTPUT_DIR:-data/interim/cninfo_expanded_5203_2018_2026}
RAW_DIR=${RAW_DIR:-data/raw/cninfo_expanded_5203_2018_2026}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-reports/cninfo_expanded_5203_2018_2026_audit.json}
FINAL_END_DATE=${FINAL_END_DATE:-2026-08-18}
CONCURRENCY=${CONCURRENCY:-20}
SNAPSHOT_FROM_TASK=${SNAPSHOT_FROM_TASK:-20260817T102235Z-simple-states-aligned-pearson-eval-v1-1750506}
EXPECTED_STOCKS=$(awk 'NR > 1 && $0 !~ /,0$/ {count++} END {print count+0}' "${STOCKS_FILE}")

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name cninfo-expanded-5203-2018-2026 \
    --purpose "Collect all CNINFO announcements for ${EXPECTED_STOCKS} return-aligned stocks from 2018 through ${FINAL_END_DATE}" \
    --input "${STOCKS_FILE}" \
    --output "${OUTPUT_DIR}" \
    --output "${RAW_DIR}" \
    --output "${AUDIT_OUTPUT}" \
    --related-file scripts/collect_cninfo_announcements.py \
    --related-file scripts/run_cninfo_paper_collection.py \
    --related-file scripts/audit_cninfo_annual_collection.py \
    --related-file scripts/slurm_cninfo_expanded_2018_2026.sbatch \
    --related-file scripts/slurm_audit_cninfo_expanded_2018_2026.sbatch \
    --related-file scripts/submit_cninfo_expanded_2018_2026.sh \
    --snapshot-from-task "${SNAPSHOT_FROM_TASK}" \
    -- bash "$0" "$@"
fi

case "${CONCURRENCY}" in
  ''|*[!0-9]*) echo "CONCURRENCY must be a positive integer" >&2; exit 2 ;;
esac
(( CONCURRENCY >= 6 && CONCURRENCY <= 32 )) || {
  echo "CONCURRENCY must be between 6 and 32" >&2
  exit 2
}

.venv/bin/python -m py_compile \
  scripts/run_cninfo_paper_collection.py \
  scripts/collect_cninfo_announcements.py \
  scripts/audit_cninfo_annual_collection.py
bash -n \
  scripts/slurm_cninfo_expanded_2018_2026.sbatch \
  scripts/slurm_audit_cninfo_expanded_2018_2026.sbatch
mkdir -p \
  logs/slurm_cninfo_expanded_2018_2026 \
  logs/slurm_cninfo_expanded_2018_2026_audit \
  "${OUTPUT_DIR}" "${RAW_DIR}"

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

chunk_count=$(((EXPECTED_STOCKS + 999) / 1000))
base_concurrency=$((CONCURRENCY / chunk_count))
extra_concurrency=$((CONCURRENCY % chunk_count))
exports="ALL,REPO_ROOT=${REPO_ROOT},STOCKS_FILE=${STOCKS_FILE},EXPECTED_STOCKS=${EXPECTED_STOCKS},OUTPUT_DIR=${OUTPUT_DIR},RAW_DIR=${RAW_DIR},FINAL_END_DATE=${FINAL_END_DATE},AUDIT_OUTPUT=${AUDIT_OUTPUT}"
collection_jobs=()
offset=0
chunk_index=0
while (( offset < EXPECTED_STOCKS )); do
  remaining=$((EXPECTED_STOCKS - offset))
  (( remaining > 1000 )) && chunk_size=1000 || chunk_size=${remaining}
  chunk_concurrency=${base_concurrency}
  (( chunk_index < extra_concurrency )) && chunk_concurrency=$((chunk_concurrency + 1))
  job=$(tracked_sbatch --parsable \
    --array="0-$((chunk_size - 1))%${chunk_concurrency}" \
    --export="${exports},STOCK_INDEX_OFFSET=${offset}" \
    scripts/slurm_cninfo_expanded_2018_2026.sbatch)
  collection_jobs+=("${job}")
  offset=$((offset + chunk_size))
  chunk_index=$((chunk_index + 1))
done
dependency=$(IFS=:; echo "${collection_jobs[*]}")
audit_job=$(tracked_sbatch --parsable \
  --dependency="afterany:${dependency}" \
  --export="${exports}" \
  scripts/slurm_audit_cninfo_expanded_2018_2026.sbatch)

printf 'collection_jobs=%s stocks=%s stock_years=%s global_concurrency=%s\n' \
  "${collection_jobs[*]}" "${EXPECTED_STOCKS}" "$((EXPECTED_STOCKS * 9))" "${CONCURRENCY}"
printf 'audit_job=%s dependency=afterany:%s\n' "${audit_job}" "${dependency}"
