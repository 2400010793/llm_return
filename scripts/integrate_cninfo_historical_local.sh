#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
WORKERS=${WORKERS:-20}
SOURCE_PANEL=${SOURCE_PANEL:-data/processed/cninfo_full_o2o_panel.parquet}
HISTORICAL_RAW=${HISTORICAL_RAW:-data/interim/cninfo_paper_816_2010_2017}
EXISTING_CLEAN=${EXISTING_CLEAN:-data/processed/cleaned/cninfo_by_file}
HISTORICAL_CLEAN=${HISTORICAL_CLEAN:-data/processed/cleaned/cninfo_historical_2010_2017_by_file}
MERGED_PREFIX=${MERGED_PREFIX:-data/processed/cleaned/cninfo_announcements_2010_2026}

# Never integrate an incomplete crawl into the modeling dataset.
"${PYTHON}" scripts/audit_cninfo_historical_collection.py \
  --stock-ids-from "${SOURCE_PANEL}" \
  --output-dir "${HISTORICAL_RAW}" \
  --start-year 2010 \
  --end-year 2017 \
  --summary reports/cninfo_historical_local_progress.json

mkdir -p "${HISTORICAL_CLEAN}"
export PYTHON HISTORICAL_CLEAN
find "${HISTORICAL_RAW}" -maxdepth 1 -type f -name '*.json' -print0 \
  | sort -z \
  | xargs -0 -r -P "${WORKERS}" -I '{}' \
      "${PYTHON}" scripts/clean_cninfo_one_file.py '{}' --output-dir "${HISTORICAL_CLEAN}"

"${PYTHON}" scripts/deduplicate_cninfo_cleaned.py \
  --input-dir "${EXISTING_CLEAN}" \
  --input-dir "${HISTORICAL_CLEAN}" \
  --stock-ids-from "${SOURCE_PANEL}" \
  --output "${MERGED_PREFIX}_deduplicated.jsonl" \
  --summary "${MERGED_PREFIX}_deduplicated.summary.json"

"${PYTHON}" scripts/finalize_cninfo_text.py \
  --input "${MERGED_PREFIX}_deduplicated.jsonl" \
  --output "${MERGED_PREFIX}_final.jsonl" \
  --excluded "${MERGED_PREFIX}_excluded_noise.jsonl" \
  --summary "${MERGED_PREFIX}_final.summary.json"
