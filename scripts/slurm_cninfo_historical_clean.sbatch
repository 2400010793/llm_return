#!/bin/bash
# Clean and merge the completed 2010--2017 CNINFO crawl with the existing
# cleaned CNINFO files. This job must run only after the historical crawl
# retry succeeds; the audit below deliberately refuses incomplete input.
#SBATCH --job-name=cninfo-clean-all
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_cninfo_clean_all/%j.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_cninfo_clean_all/%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?historical CNINFO cleaning must use scripts/task_tracker.py}"

PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
SOURCE_PANEL=${SOURCE_PANEL:-data/processed/cninfo_full_o2o_panel.parquet}
RAW_DIR=${RAW_DIR:-data/interim/cninfo_paper_816_2010_2017}
HISTORICAL_CLEAN=${HISTORICAL_CLEAN:-data/processed/cleaned/cninfo_historical_2010_2017_by_file}
EXISTING_CLEAN=${EXISTING_CLEAN:-data/processed/cleaned/cninfo_by_file}
MERGED_PREFIX=${MERGED_PREFIX:-data/processed/cleaned/cninfo_announcements_2010_2026}
AUDIT_SUMMARY=${AUDIT_SUMMARY:-reports/cninfo_historical_cleaning_audit.json}
ALLOW_INCOMPLETE_HISTORICAL=${ALLOW_INCOMPLETE_HISTORICAL:-0}

mkdir -p logs/slurm_cninfo_clean_all "${HISTORICAL_CLEAN}" data/processed/cleaned
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# The default remains strict. A deliberately provisional run may opt in to
# cleaning the available complete shards, but its output must use a separate
# prefix and must not be presented as complete 2010--2017 coverage.
if ! "${PYTHON}" scripts/audit_cninfo_historical_collection.py \
  --stock-ids-from "${SOURCE_PANEL}" \
  --output-dir "${RAW_DIR}" \
  --start-year 2010 \
  --end-year 2017 \
  --summary "${AUDIT_SUMMARY}"; then
  if [[ "${ALLOW_INCOMPLETE_HISTORICAL}" != "1" ]]; then
    echo "Historical crawl is incomplete; refusing to build a merged dataset" >&2
    exit 1
  fi
  echo "WARNING: continuing with incomplete historical coverage; output is provisional" >&2
fi

mapfile -t raw_files < <(find "${RAW_DIR}" -maxdepth 1 -type f -name '*.json' -print | sort)
if (( ${#raw_files[@]} == 0 )); then
  echo "No historical CNINFO JSON files found in ${RAW_DIR}" >&2
  exit 1
fi

# One output file and one summary per stock-year make the cleaning stage
# restartable and prevent a single malformed record from losing all progress.
printf '%s\0' "${raw_files[@]}" \
  | xargs -0 -r -P "${SLURM_CPUS_PER_TASK:-16}" -I '{}' \
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

test -s "${MERGED_PREFIX}_final.jsonl"
echo "Historical CNINFO cleaning and merge completed: ${MERGED_PREFIX}_final.jsonl"