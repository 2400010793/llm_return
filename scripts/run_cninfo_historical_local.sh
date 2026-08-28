#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"

SOURCE_PANEL=${SOURCE_PANEL:-data/processed/cninfo_full_o2o_panel.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-data/interim/cninfo_paper_816_2010_2017}
RAW_DIR=${RAW_DIR:-data/raw/cninfo_paper_816_2010_2017}
WORKERS=${WORKERS:-4}
MAX_PASSES=${MAX_PASSES:-3}
LOCK_FILE=${LOCK_FILE:-logs/cninfo_historical_local.lock}
PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}

mkdir -p logs "${OUTPUT_DIR}" "${RAW_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo '{"status":"already_running","lock":"'"${LOCK_FILE}"'"}' >&2
  exit 2
fi

overall_status=0
for YEAR in $(seq 2010 2017); do
  for PASS in $(seq 1 "${MAX_PASSES}"); do
    echo '{"event":"year_pass_start","year":'"${YEAR}"',"pass":'"${PASS}"',"workers":'"${WORKERS}"'}'
    if "${PYTHON}" scripts/run_cninfo_paper_collection.py \
      --stocks data/stock_universe_paper_1000.csv \
      --stock-ids-from "${SOURCE_PANEL}" \
      --expected-stocks 816 \
      --start-date "${YEAR}-01-01" \
      --end-date "${YEAR}-12-31" \
      --limit 0 \
      --workers "${WORKERS}" \
      --stock-timeout-seconds 5400 \
      --resolve-stock \
      --max-pages 0 \
      --page-pause-seconds 2 \
      --pause-seconds 1 \
      --stock-pause-seconds 0 \
      --output-dir "${OUTPUT_DIR}" \
      --output-suffix "${YEAR}" \
      --raw-dir "${RAW_DIR}"; then
      echo '{"event":"year_complete","year":'"${YEAR}"',"pass":'"${PASS}"'}'
      break
    fi
    overall_status=1
    echo '{"event":"year_partial","year":'"${YEAR}"',"pass":'"${PASS}"'}' >&2
  done
done

"${PYTHON}" scripts/audit_cninfo_historical_collection.py \
  --stock-ids-from "${SOURCE_PANEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --start-year 2010 \
  --end-year 2017 \
  --summary reports/cninfo_historical_local_progress.json
audit_status=$?
if (( audit_status != 0 )); then
  overall_status=1
fi
exit "${overall_status}"
